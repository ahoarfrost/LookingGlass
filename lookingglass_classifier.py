"""
LookingGlass Classifiers - Fine-tuned DNA sequence classifiers

Pure PyTorch implementation of LookingGlass classifiers from the paper.
Uses LookingGlass encoder with classification head.

Usage:
    from lookingglass_classifier import LookingGlassClassifier, LookingGlassTokenizer

    model = LookingGlassClassifier.from_pretrained('.')
    tokenizer = LookingGlassTokenizer()

    inputs = tokenizer(["GATTACA"], return_tensors=True)
    logits = model(inputs['input_ids'])  # (batch, num_classes)
    predictions = logits.argmax(dim=-1)
"""

import json
import os
from dataclasses import dataclass, asdict, field
from typing import Optional, List

import torch
import torch.nn as nn

from lookingglass import (
    LookingGlassConfig,
    LookingGlassTokenizer,
    _AWDLSTMEncoder,
    _is_hf_hub_id,
    _download_from_hub,
)

__version__ = "1.1.0"
__all__ = ["LookingGlassClassifierConfig", "LookingGlassClassifier", "LookingGlassTokenizer"]


@dataclass
class LookingGlassClassifierConfig(LookingGlassConfig):
    """Configuration for LookingGlass classifier."""
    num_classes: int = 2
    classifier_hidden: int = 50
    classifier_dropout: float = 0.0
    class_names: List[str] = field(default_factory=list)

    def save_pretrained(self, save_directory: str):
        os.makedirs(save_directory, exist_ok=True)
        with open(os.path.join(save_directory, "config.json"), 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_pretrained(cls, pretrained_path: str) -> "LookingGlassClassifierConfig":
        if _is_hf_hub_id(pretrained_path):
            try:
                config_path = _download_from_hub(pretrained_path, "config.json")
            except Exception:
                return cls()
        elif os.path.isdir(pretrained_path):
            config_path = os.path.join(pretrained_path, "config.json")
        else:
            config_path = pretrained_path

        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
            return cls(**{k: v for k, v in config_dict.items() if k in valid_fields})
        return cls()


class LookingGlassClassifier(nn.Module):
    """
    LookingGlass with classification head.

    Uses concat pooling (max + mean + last) followed by classification layers.

    Example:
        >>> model = LookingGlassClassifier.from_pretrained('.')
        >>> tokenizer = LookingGlassTokenizer()
        >>> inputs = tokenizer("GATTACA", return_tensors=True)
        >>> logits = model(inputs['input_ids'])  # (1, num_classes)
        >>> prediction = logits.argmax(dim=-1)
    """

    def __init__(self, config: Optional[LookingGlassClassifierConfig] = None):
        super().__init__()
        self.config = config or LookingGlassClassifierConfig()
        self.encoder = _AWDLSTMEncoder(self.config)

        # Concat pooling: max + mean + last = 3 * hidden_size
        pooled_size = 3 * self.config.hidden_size

        # Classification head: BatchNorm -> Linear -> ReLU -> BatchNorm -> Linear
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(pooled_size),
            nn.Dropout(self.config.classifier_dropout),
            nn.Linear(pooled_size, self.config.classifier_hidden),
            nn.ReLU(),
            nn.BatchNorm1d(self.config.classifier_hidden),
            nn.Dropout(self.config.classifier_dropout),
            nn.Linear(self.config.classifier_hidden, self.config.num_classes),
        )

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """
        Forward pass returning classification logits.

        Args:
            input_ids: Token indices (batch, seq_len)

        Returns:
            Logits (batch, num_classes)
        """
        self.encoder.reset()
        hidden = self.encoder(input_ids)  # (batch, seq_len, hidden_size)

        # Concat pooling: max, mean, last
        max_pool = hidden.max(dim=1).values
        mean_pool = hidden.mean(dim=1)
        last_pool = hidden[:, -1]
        pooled = torch.cat([max_pool, mean_pool, last_pool], dim=-1)

        return self.classifier(pooled)

    def predict(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """Return predicted class indices."""
        logits = self.forward(input_ids)
        return logits.argmax(dim=-1)

    def predict_proba(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """Return class probabilities."""
        logits = self.forward(input_ids)
        return torch.softmax(logits, dim=-1)

    def get_embeddings(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """Get sequence embeddings (last token) from encoder."""
        self.encoder.reset()
        hidden = self.encoder(input_ids)
        return hidden[:, -1]

    def save_pretrained(self, save_directory: str):
        os.makedirs(save_directory, exist_ok=True)
        self.config.save_pretrained(save_directory)
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))

    @classmethod
    def from_pretrained(
        cls, pretrained_path: str, config: Optional[LookingGlassClassifierConfig] = None
    ) -> "LookingGlassClassifier":
        config = config or LookingGlassClassifierConfig.from_pretrained(pretrained_path)
        model = cls(config)

        if _is_hf_hub_id(pretrained_path):
            model_path = _download_from_hub(pretrained_path, "pytorch_model.bin")
        else:
            model_path = os.path.join(pretrained_path, "pytorch_model.bin")

        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)

        return model


def convert_classifier_weights(
    original_path: str,
    output_dir: str,
    num_classes: int,
    class_names: Optional[List[str]] = None,
):
    """
    Convert original fastai classifier weights to pure PyTorch format.

    Args:
        original_path: Path to original .pth file
        output_dir: Output directory for converted model
        num_classes: Number of output classes
        class_names: Optional list of class names
    """
    print(f"Loading weights from {original_path}...")
    original = torch.load(original_path, map_location='cpu')
    if 'model' in original:
        original = original['model']

    # Create config
    config = LookingGlassClassifierConfig(
        num_classes=num_classes,
        classifier_hidden=50,
        class_names=class_names or [],
    )

    # Create model
    model = LookingGlassClassifier(config)

    # Map weights
    new_state = {}

    # Encoder weights
    weight_map = {
        '0.module.encoder.weight': 'encoder.embed_tokens.weight',
        '0.module.encoder_dp.emb.weight': 'encoder.embed_dropout.embedding.weight',
    }

    for i in range(3):
        weight_map.update({
            f'0.module.rnns.{i}.weight_hh_l0_raw': f'encoder.layers.{i}.weight_hh_l0_raw',
            f'0.module.rnns.{i}.module.weight_ih_l0': f'encoder.layers.{i}.module.weight_ih_l0',
            f'0.module.rnns.{i}.module.weight_hh_l0': f'encoder.layers.{i}.module.weight_hh_l0',
            f'0.module.rnns.{i}.module.bias_ih_l0': f'encoder.layers.{i}.module.bias_ih_l0',
            f'0.module.rnns.{i}.module.bias_hh_l0': f'encoder.layers.{i}.module.bias_hh_l0',
        })

    # Classifier head weights
    # Original: 1.layers.{0,2,4,6} -> our Sequential indices
    classifier_map = {
        '1.layers.0.weight': 'classifier.0.weight',
        '1.layers.0.bias': 'classifier.0.bias',
        '1.layers.0.running_mean': 'classifier.0.running_mean',
        '1.layers.0.running_var': 'classifier.0.running_var',
        '1.layers.0.num_batches_tracked': 'classifier.0.num_batches_tracked',
        '1.layers.2.weight': 'classifier.2.weight',
        '1.layers.2.bias': 'classifier.2.bias',
        '1.layers.4.weight': 'classifier.4.weight',
        '1.layers.4.bias': 'classifier.4.bias',
        '1.layers.4.running_mean': 'classifier.4.running_mean',
        '1.layers.4.running_var': 'classifier.4.running_var',
        '1.layers.4.num_batches_tracked': 'classifier.4.num_batches_tracked',
        '1.layers.6.weight': 'classifier.6.weight',
        '1.layers.6.bias': 'classifier.6.bias',
    }
    weight_map.update(classifier_map)

    for old_key, new_key in weight_map.items():
        if old_key in original:
            new_state[new_key] = original[old_key]

    # Load and save
    model.load_state_dict(new_state, strict=False)

    os.makedirs(output_dir, exist_ok=True)
    config.save_pretrained(output_dir)
    torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))

    print(f"Saved to {output_dir}")
    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert LookingGlass classifier weights")
    parser.add_argument("--input", required=True, help="Path to original .pth file")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--num-classes", type=int, required=True, help="Number of classes")
    parser.add_argument("--class-names", nargs="+", help="Class names")

    args = parser.parse_args()
    convert_classifier_weights(args.input, args.output, args.num_classes, args.class_names)
