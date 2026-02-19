"""
LookingGlass - A DNA Language Model

Pure PyTorch implementation of LookingGlass, a pretrained language model for DNA sequences.
Based on AWD-LSTM architecture, originally trained with fastai v1.

Paper: Hoarfrost et al., "Deep learning of a bacterial and archaeal universal language
of life enables transfer learning and illuminates microbial dark matter",
Nature Communications, 2022.

Usage:
    from lookingglass import LookingGlass, LookingGlassTokenizer

    # Load from HuggingFace Hub
    model = LookingGlass.from_pretrained('HoarfrostLab/lookingglass-v1')
    tokenizer = LookingGlassTokenizer()

    # Or load from local path
    model = LookingGlass.from_pretrained('./lookingglass-v1')

    inputs = tokenizer(["GATTACA", "ATCGATCG"], return_tensors=True)
    embeddings = model.get_embeddings(inputs['input_ids'])  # (batch, 104)
"""

import json
import os
import warnings
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List, Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from huggingface_hub import hf_hub_download
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False


__version__ = "1.1.0"


def _is_hf_hub_id(path: str) -> bool:
    """Check if path looks like a HuggingFace Hub model ID (e.g., 'user/model')."""
    if os.path.exists(path):
        return False
    return '/' in path and not path.startswith(('.', '/'))


def _download_from_hub(repo_id: str, filename: str) -> str:
    """Download a file from HuggingFace Hub and return the local path."""
    if not HF_HUB_AVAILABLE:
        raise ImportError(
            "huggingface_hub is required to load models from the Hub. "
            "Install it with: pip install huggingface_hub"
        )
    return hf_hub_download(repo_id=repo_id, filename=filename)
__all__ = [
    "LookingGlassConfig",
    "LookingGlass",
    "LookingGlassLM",
    "LookingGlassTokenizer",
]


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class LookingGlassConfig:
    """
    Configuration for LookingGlass model.

    Default values match the original pretrained LookingGlass model.
    """
    vocab_size: int = 8
    hidden_size: int = 104          # embedding/output size
    intermediate_size: int = 1152   # LSTM hidden size
    num_hidden_layers: int = 3
    pad_token_id: int = 1
    bos_token_id: int = 2
    eos_token_id: int = 3
    bidirectional: bool = False     # original LG is unidirectional
    output_dropout: float = 0.1
    hidden_dropout: float = 0.15
    input_dropout: float = 0.25
    embed_dropout: float = 0.02
    weight_dropout: float = 0.2
    tie_weights: bool = True
    output_bias: bool = True
    model_type: str = "lookingglass"

    def to_dict(self) -> Dict:
        return asdict(self)

    def save_pretrained(self, save_directory: str):
        os.makedirs(save_directory, exist_ok=True)
        with open(os.path.join(save_directory, "config.json"), 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_pretrained(cls, pretrained_path: str) -> "LookingGlassConfig":
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


# =============================================================================
# Tokenizer
# =============================================================================

VOCAB = ['xxunk', 'xxpad', 'xxbos', 'xxeos', 'G', 'A', 'C', 'T']
VOCAB_TO_ID = {tok: i for i, tok in enumerate(VOCAB)}
ID_TO_VOCAB = {i: tok for i, tok in enumerate(VOCAB)}


class LookingGlassTokenizer:
    """
    Tokenizer for DNA sequences.

    Each nucleotide (G, A, C, T) is a single token. By default, adds BOS token
    at the start of each sequence (matching original LookingGlass training).

    Special tokens:
        - xxunk (0): Unknown
        - xxpad (1): Padding
        - xxbos (2): Beginning of sequence
        - xxeos (3): End of sequence
    """

    vocab = VOCAB
    vocab_to_id = VOCAB_TO_ID
    id_to_vocab = ID_TO_VOCAB

    def __init__(
        self,
        add_bos_token: bool = True,   # original LG uses BOS
        add_eos_token: bool = False,  # original LG does not use EOS
        padding_side: str = "right",
    ):
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token
        self.padding_side = padding_side

        self.unk_token_id = 0
        self.pad_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, sequence: str, add_special_tokens: bool = True) -> List[int]:
        """Encode a DNA sequence to token IDs."""
        tokens = []

        if add_special_tokens and self.add_bos_token:
            tokens.append(self.bos_token_id)

        for char in sequence.upper():
            if char in self.vocab_to_id:
                tokens.append(self.vocab_to_id[char])
            elif char.strip():
                tokens.append(self.unk_token_id)

        if add_special_tokens and self.add_eos_token:
            tokens.append(self.eos_token_id)

        return tokens

    def decode(self, token_ids: Union[List[int], torch.Tensor], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to DNA sequence."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        special_ids = {0, 1, 2, 3}
        tokens = []
        for tid in token_ids:
            if skip_special_tokens and tid in special_ids:
                continue
            tokens.append(self.id_to_vocab.get(tid, 'xxunk'))
        return ''.join(tokens)

    def __call__(
        self,
        sequences: Union[str, List[str]],
        padding: Union[bool, str] = False,
        max_length: Optional[int] = None,
        truncation: bool = False,
        return_tensors: Union[bool, str] = False,
        return_attention_mask: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Tokenize DNA sequence(s)."""
        if isinstance(sequences, str):
            sequences = [sequences]
            single = True
        else:
            single = False

        encoded = [self.encode(seq) for seq in sequences]

        if truncation and max_length:
            encoded = [e[:max_length] for e in encoded]

        # Padding
        if padding or len(encoded) > 1:
            if padding == 'max_length' and max_length:
                pad_len = max_length
            else:
                pad_len = max(len(e) for e in encoded)

            padded = []
            masks = []
            for e in encoded:
                pad_amount = pad_len - len(e)
                mask = [1] * len(e) + [0] * pad_amount
                if self.padding_side == 'right':
                    e = e + [self.pad_token_id] * pad_amount
                else:
                    e = [self.pad_token_id] * pad_amount + e
                    mask = [0] * pad_amount + [1] * len(e)
                padded.append(e)
                masks.append(mask)
            encoded = padded
        else:
            masks = [[1] * len(e) for e in encoded]

        result = {}
        if return_tensors in ('pt', True):
            result['input_ids'] = torch.tensor(encoded, dtype=torch.long)
            if return_attention_mask:
                result['attention_mask'] = torch.tensor(masks, dtype=torch.long)
        else:
            result['input_ids'] = encoded[0] if single else encoded
            if return_attention_mask:
                result['attention_mask'] = masks[0] if single else masks

        return result

    def save_pretrained(self, save_directory: str):
        os.makedirs(save_directory, exist_ok=True)
        with open(os.path.join(save_directory, "vocab.json"), 'w') as f:
            json.dump(self.vocab_to_id, f, indent=2)
        with open(os.path.join(save_directory, "tokenizer_config.json"), 'w') as f:
            json.dump({
                "add_bos_token": self.add_bos_token,
                "add_eos_token": self.add_eos_token,
                "padding_side": self.padding_side,
            }, f, indent=2)

    @classmethod
    def from_pretrained(cls, pretrained_path: str) -> "LookingGlassTokenizer":
        kwargs = {}
        if _is_hf_hub_id(pretrained_path):
            try:
                config_path = _download_from_hub(pretrained_path, "tokenizer_config.json")
                with open(config_path, 'r') as f:
                    kwargs = json.load(f)
            except Exception:
                pass
        else:
            config_path = os.path.join(pretrained_path, "tokenizer_config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    kwargs = json.load(f)
        return cls(**kwargs)


# =============================================================================
# Model Components
# =============================================================================

def _dropout_mask(x: torch.Tensor, size: Tuple[int, ...], p: float) -> torch.Tensor:
    """Create dropout mask with inverted scaling."""
    return x.new_empty(*size).bernoulli_(1 - p).div_(1 - p)


class _RNNDropout(nn.Module):
    """Dropout consistent across sequence dimension."""

    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.:
            return x
        mask = _dropout_mask(x.data, (x.size(0), 1, x.size(2)), self.p)
        return x * mask


class _EmbeddingDropout(nn.Module):
    """Dropout applied to entire embedding rows."""

    def __init__(self, embedding: nn.Embedding, p: float):
        super().__init__()
        self.embedding = embedding
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.p != 0:
            mask = _dropout_mask(self.embedding.weight.data,
                                 (self.embedding.weight.size(0), 1), self.p)
            masked_weight = self.embedding.weight * mask
        else:
            masked_weight = self.embedding.weight

        padding_idx = self.embedding.padding_idx if self.embedding.padding_idx is not None else -1
        return F.embedding(x, masked_weight, padding_idx,
                          self.embedding.max_norm, self.embedding.norm_type,
                          self.embedding.scale_grad_by_freq, self.embedding.sparse)


class _WeightDropout(nn.Module):
    """DropConnect applied to RNN hidden-to-hidden weights."""

    def __init__(self, module: nn.Module, p: float, layer_names='weight_hh_l0'):
        super().__init__()
        self.module = module
        self.p = p
        self.layer_names = [layer_names] if isinstance(layer_names, str) else layer_names

        for layer in self.layer_names:
            w = getattr(self.module, layer)
            delattr(self.module, layer)
            self.register_parameter(f'{layer}_raw', nn.Parameter(w.data))
            setattr(self.module, layer, w.clone())

        if isinstance(self.module, nn.RNNBase):
            self.module.flatten_parameters = lambda: None

    def _set_weights(self):
        for layer in self.layer_names:
            raw_w = getattr(self, f'{layer}_raw')
            w = F.dropout(raw_w, p=self.p, training=self.training) if self.training else raw_w.clone()
            setattr(self.module, layer, w)

    def forward(self, *args):
        self._set_weights()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            return self.module(*args)


class _AWDLSTMEncoder(nn.Module):
    """AWD-LSTM encoder backbone."""

    _init_range = 0.1

    def __init__(self, config: LookingGlassConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.num_layers = config.num_hidden_layers
        self.num_directions = 2 if config.bidirectional else 1
        self._batch_size = 1

        # Embedding
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size,
                                         padding_idx=config.pad_token_id)
        self.embed_tokens.weight.data.uniform_(-self._init_range, self._init_range)
        self.embed_dropout = _EmbeddingDropout(self.embed_tokens, config.embed_dropout)

        # LSTM layers
        self.layers = nn.ModuleList()
        for i in range(config.num_hidden_layers):
            input_size = config.hidden_size if i == 0 else config.intermediate_size
            output_size = (config.intermediate_size if i != config.num_hidden_layers - 1
                          else config.hidden_size) // self.num_directions
            lstm = nn.LSTM(input_size, output_size, num_layers=1,
                          batch_first=True, bidirectional=config.bidirectional)
            self.layers.append(_WeightDropout(lstm, config.weight_dropout))

        # Dropout
        self.input_dropout = _RNNDropout(config.input_dropout)
        self.hidden_dropout = nn.ModuleList([
            _RNNDropout(config.hidden_dropout) for _ in range(config.num_hidden_layers)
        ])

        self._hidden_state = None
        self.reset()

    def reset(self):
        """Reset LSTM hidden states."""
        self._hidden_state = [self._init_hidden(i) for i in range(self.num_layers)]

    def _init_hidden(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        nh = (self.intermediate_size if layer_idx != self.num_layers - 1
              else self.hidden_size) // self.num_directions
        weight = next(self.parameters())
        return (weight.new_zeros(self.num_directions, self._batch_size, nh),
                weight.new_zeros(self.num_directions, self._batch_size, nh))

    def _resize_hidden(self, batch_size: int):
        new_hidden = []
        for i in range(self.num_layers):
            nh = (self.intermediate_size if i != self.num_layers - 1
                  else self.hidden_size) // self.num_directions
            h, c = self._hidden_state[i]

            if self._batch_size < batch_size:
                h = torch.cat([h, h.new_zeros(self.num_directions, batch_size - self._batch_size, nh)], dim=1)
                c = torch.cat([c, c.new_zeros(self.num_directions, batch_size - self._batch_size, nh)], dim=1)
            elif self._batch_size > batch_size:
                h = h[:, :batch_size].contiguous()
                c = c[:, :batch_size].contiguous()
            new_hidden.append((h, c))

        self._hidden_state = new_hidden
        self._batch_size = batch_size

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """Returns hidden states for all positions: (batch, seq_len, hidden_size)"""
        batch_size, seq_len = input_ids.shape

        if batch_size != self._batch_size:
            self._resize_hidden(batch_size)

        hidden = self.input_dropout(self.embed_dropout(input_ids))

        new_hidden = []
        for i, (layer, hdp) in enumerate(zip(self.layers, self.hidden_dropout)):
            hidden, h = layer(hidden, self._hidden_state[i])
            new_hidden.append(h)
            if i != self.num_layers - 1:
                hidden = hdp(hidden)

        self._hidden_state = [(h.detach(), c.detach()) for h, c in new_hidden]
        return hidden


class _LMHead(nn.Module):
    """Language modeling head."""

    _init_range = 0.1

    def __init__(self, config: LookingGlassConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__()
        self.output_dropout = _RNNDropout(config.output_dropout)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=config.output_bias)
        self.decoder.weight.data.uniform_(-self._init_range, self._init_range)

        if config.output_bias:
            self.decoder.bias.data.zero_()

        if embed_tokens is not None and config.tie_weights:
            self.decoder.weight = embed_tokens.weight

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.output_dropout(hidden_states))


# =============================================================================
# Models
# =============================================================================

class LookingGlass(nn.Module):
    """
    LookingGlass encoder model.

    Outputs sequence embeddings for downstream tasks (classification, clustering, etc.).
    Uses last-token embedding by default, matching original LookingGlass.

    Example:
        >>> model = LookingGlass.from_pretrained('lookingglass-v1')
        >>> tokenizer = LookingGlassTokenizer()
        >>> inputs = tokenizer("GATTACA", return_tensors=True)
        >>> embeddings = model.get_embeddings(inputs['input_ids'])  # (1, 104)
    """

    config_class = LookingGlassConfig

    def __init__(self, config: Optional[LookingGlassConfig] = None):
        super().__init__()
        self.config = config or LookingGlassConfig()
        self.encoder = _AWDLSTMEncoder(self.config)

    def reset(self):
        """Reset hidden states."""
        self.encoder.reset()

    def forward(self, input_ids: torch.LongTensor, **kwargs) -> torch.Tensor:
        """
        Forward pass. Returns last-token embeddings.

        Args:
            input_ids: Token indices (batch, seq_len)

        Returns:
            Embeddings (batch, hidden_size)
        """
        return self.get_embeddings(input_ids)

    def get_embeddings(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """
        Get sequence embeddings using last-token pooling (original LG method).

        Resets hidden state before encoding for deterministic results.

        Args:
            input_ids: Token indices (batch, seq_len)

        Returns:
            Embeddings (batch, hidden_size)
        """
        self.encoder.reset()
        hidden = self.encoder(input_ids)  # (batch, seq_len, hidden_size)
        return hidden[:, -1]  # last token

    def get_hidden_states(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """
        Get hidden states for all positions.

        Resets hidden state before encoding for deterministic results.

        Args:
            input_ids: Token indices (batch, seq_len)

        Returns:
            Hidden states (batch, seq_len, hidden_size)
        """
        self.encoder.reset()
        return self.encoder(input_ids)

    def save_pretrained(self, save_directory: str):
        os.makedirs(save_directory, exist_ok=True)
        self.config.save_pretrained(save_directory)
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))

    @classmethod
    def from_pretrained(cls, pretrained_path: str, config: Optional[LookingGlassConfig] = None) -> "LookingGlass":
        config = config or LookingGlassConfig.from_pretrained(pretrained_path)
        model = cls(config)

        if _is_hf_hub_id(pretrained_path):
            model_path = _download_from_hub(pretrained_path, "pytorch_model.bin")
        else:
            model_path = os.path.join(pretrained_path, "pytorch_model.bin")

        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location='cpu')
            # Only load encoder weights
            encoder_state_dict = {k: v for k, v in state_dict.items()
                                  if not k.startswith('lm_head.')}
            model.load_state_dict(encoder_state_dict, strict=False)

        return model


class LookingGlassLM(nn.Module):
    """
    LookingGlass with language modeling head.

    Full model for next-token prediction. Can also extract embeddings.

    Example:
        >>> model = LookingGlassLM.from_pretrained('lookingglass-v1')
        >>> tokenizer = LookingGlassTokenizer()
        >>> inputs = tokenizer("GATTACA", return_tensors=True)
        >>> logits = model(inputs['input_ids'])  # (1, 8, 8)
        >>> embeddings = model.get_embeddings(inputs['input_ids'])  # (1, 104)
    """

    config_class = LookingGlassConfig

    def __init__(self, config: Optional[LookingGlassConfig] = None):
        super().__init__()
        self.config = config or LookingGlassConfig()
        self.encoder = _AWDLSTMEncoder(self.config)
        self.lm_head = _LMHead(
            self.config,
            embed_tokens=self.encoder.embed_tokens if self.config.tie_weights else None
        )

    def reset(self):
        """Reset hidden states."""
        self.encoder.reset()

    def forward(self, input_ids: torch.LongTensor, **kwargs) -> torch.Tensor:
        """
        Forward pass. Returns logits for next-token prediction.

        Args:
            input_ids: Token indices (batch, seq_len)

        Returns:
            Logits (batch, seq_len, vocab_size)
        """
        hidden = self.encoder(input_ids)
        return self.lm_head(hidden)

    def get_embeddings(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """
        Get sequence embeddings using last-token pooling.

        Resets hidden state before encoding for deterministic results.

        Args:
            input_ids: Token indices (batch, seq_len)

        Returns:
            Embeddings (batch, hidden_size)
        """
        self.encoder.reset()
        hidden = self.encoder(input_ids)
        return hidden[:, -1]

    def get_hidden_states(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """
        Get hidden states for all positions.

        Resets hidden state before encoding for deterministic results.

        Args:
            input_ids: Token indices (batch, seq_len)

        Returns:
            Hidden states (batch, seq_len, hidden_size)
        """
        self.encoder.reset()
        return self.encoder(input_ids)

    def save_pretrained(self, save_directory: str):
        os.makedirs(save_directory, exist_ok=True)
        self.config.save_pretrained(save_directory)
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))

    @classmethod
    def from_pretrained(cls, pretrained_path: str, config: Optional[LookingGlassConfig] = None) -> "LookingGlassLM":
        config = config or LookingGlassConfig.from_pretrained(pretrained_path)
        model = cls(config)

        if _is_hf_hub_id(pretrained_path):
            model_path = _download_from_hub(pretrained_path, "pytorch_model.bin")
        else:
            model_path = os.path.join(pretrained_path, "pytorch_model.bin")

        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)

        return model


# =============================================================================
# Weight Loading
# =============================================================================

def load_original_weights(model: Union[LookingGlass, LookingGlassLM], weights_path: str) -> None:
    """
    Load weights from original fastai-trained LookingGlass checkpoint.

    Args:
        model: Model to load weights into
        weights_path: Path to LookingGlass.pth or LookingGlass_enc.pth
    """
    checkpoint = torch.load(weights_path, map_location='cpu')

    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    is_lm_model = isinstance(model, LookingGlassLM)

    new_state_dict = {}
    for k, v in state_dict.items():
        if '.module.weight_hh_l0' in k:
            continue

        if k.startswith('0.'):
            new_k = k[2:]
            new_k = new_k.replace('encoder.', 'embed_tokens.')
            new_k = new_k.replace('encoder_dp.emb.', 'embed_tokens.')
            new_k = new_k.replace('rnns.', 'layers.')
            new_k = new_k.replace('hidden_dps.', 'hidden_dropout.')
            new_k = new_k.replace('input_dp.', 'input_dropout.')
            new_state_dict['encoder.' + new_k] = v

        elif k.startswith('1.') and is_lm_model:
            new_k = k[2:]
            new_k = new_k.replace('output_dp.', 'output_dropout.')
            new_state_dict['lm_head.' + new_k] = v

        else:
            new_k = k.replace('encoder.', 'embed_tokens.')
            new_k = new_k.replace('encoder_dp.emb.', 'embed_tokens.')
            new_k = new_k.replace('rnns.', 'layers.')
            new_k = new_k.replace('hidden_dps.', 'hidden_dropout.')
            new_k = new_k.replace('input_dp.', 'input_dropout.')
            new_state_dict['encoder.' + new_k] = v

    model.load_state_dict(new_state_dict, strict=False)


def convert_checkpoint(input_path: str, output_dir: str) -> None:
    """Convert original checkpoint to new format."""
    config = LookingGlassConfig()
    model = LookingGlassLM(config)
    load_original_weights(model, input_path)
    model.save_pretrained(output_dir)

    tokenizer = LookingGlassTokenizer()
    tokenizer.save_pretrained(output_dir)
    print(f"Saved to {output_dir}")


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='LookingGlass DNA Language Model')
    parser.add_argument('--convert', type=str, help='Convert original weights')
    parser.add_argument('--output', type=str, default='./lookingglass-v1', help='Output directory')
    parser.add_argument('--test', action='store_true', help='Run tests')
    args = parser.parse_args()

    if args.convert:
        convert_checkpoint(args.convert, args.output)

    elif args.test:
        print("Testing LookingGlass...\n")

        tokenizer = LookingGlassTokenizer()
        print(f"Vocab: {tokenizer.vocab}")
        print(f"BOS token added: {tokenizer.add_bos_token}")
        print(f"EOS token added: {tokenizer.add_eos_token}")

        inputs = tokenizer("GATTACA", return_tensors=True)
        print(f"\nTokenized 'GATTACA': {inputs['input_ids']}")
        print(f"Decoded: {tokenizer.decode(inputs['input_ids'][0])}")

        config = LookingGlassConfig()
        print(f"\nConfig: bidirectional={config.bidirectional}")

        # Test LookingGlass (encoder)
        encoder = LookingGlass(config)
        print(f"\nLookingGlass params: {sum(p.numel() for p in encoder.parameters()):,}")

        encoder.eval()
        with torch.no_grad():
            emb = encoder.get_embeddings(inputs['input_ids'])
            print(f"Embeddings shape: {emb.shape}")

        # Test LookingGlassLM
        lm = LookingGlassLM(config)
        print(f"\nLookingGlassLM params: {sum(p.numel() for p in lm.parameters()):,}")

        lm.eval()
        with torch.no_grad():
            logits = lm(inputs['input_ids'])
            emb = lm.get_embeddings(inputs['input_ids'])
            print(f"Logits shape: {logits.shape}")
            print(f"Embeddings shape: {emb.shape}")

        print("\nAll tests passed!")

    else:
        parser.print_help()
