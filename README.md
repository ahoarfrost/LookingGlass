# Welcome to LookingGlass

LookingGlass is a general-purpose 'universal language of life' deep learning model for read-length biological sequences. It can be used for diverse downstream transfer learning tasks for biological data, some of which are described in the paper.

This is the main repository for these pretrained models. **Release v1.1** provides a pure PyTorch implementation with no fastai dependencies. Static URLs for downloading these models are available in [release v1.1](https://github.com/ahoarfrost/LookingGlass/releases/tag/v1.1) of this repo. They can also be accessed on [HuggingFace at HoarfrostLab/lookingglass-v1](https://huggingface.co/HoarfrostLab/lookingglass-v1).

> **Note:** Looking for the original fastai-based models? See [release v1.0](https://github.com/ahoarfrost/LookingGlass/releases/tag/v1.0) and the associated [fastBio](https://github.com/ahoarfrost/fastBio) repository.

If you find LookingGlass, LookingGlass-derived models, or fastBio helpful, please cite the paper:

> Hoarfrost, A., Aptekmann, A., Farfañuk, G. et al. Deep learning of a bacterial and archaeal universal language of life enables transfer learning and illuminates microbial dark matter. Nat Commun 13, 2606 (2022). https://doi.org/10.1038/s41467-022-30070-8.

# Models in the most recent release

The most recent [release of LookingGlass (v1.1)](https://github.com/ahoarfrost/LookingGlass/releases/tag/v1.1) provides a pure PyTorch implementation with no external dependencies beyond PyTorch.

The model is also available on HuggingFace Hub: [HoarfrostLab/lookingglass-v1](https://huggingface.co/HoarfrostLab/lookingglass-v1)

* **LookingGlass** ([HuggingFace](https://huggingface.co/HoarfrostLab/lookingglass-v1))

    LookingGlass is a 'universal language of life', producing contextually-aware, functionally and evolutionarily relevant representations of short DNA reads. As a general purpose 'biological language' representation model, it is broadly useful for training diverse downstream transfer learning tasks.

## Fine-tuned Classifiers

The following classifiers are fine-tuned from the LookingGlass base model:

* **Functional Classifier** ([HuggingFace](https://huggingface.co/HoarfrostLab/LGv1_FunctionalClassifier))

    Classifies DNA reads into one of 1274 experimentally-validated functional annotations with 81.5% accuracy.

* **Optimal Temperature Classifier** ([HuggingFace](https://huggingface.co/HoarfrostLab/LGv1_OptimalTempClassifier))

    Identifies whether a DNA read originates from an enzyme with psychrophilic (<15°C), mesophilic (20-40°C), or thermophilic (>50°C) optimal temperature with 70.1% accuracy.

* **Oxidoreductase Classifier** ([HuggingFace](https://huggingface.co/HoarfrostLab/LGv1_OxidoreductaseClassifier))

    Classifies whether a DNA read originates from an oxidoreductase (EC 1.-.-.-) with 82.3% accuracy.

* **Reading Frame Classifier** ([HuggingFace](https://huggingface.co/HoarfrostLab/LGv1_ReadingFrameClassifier))

    Identifies the correct reading frame (1, 2, 3, -1, -2, -3) for prokaryotic DNA reads.

## Model Architecture

| | |
|---|---|
| Architecture | AWD-LSTM (3-layer, unidirectional) |
| Hidden size | 1152 |
| Embedding size | 104 |
| Parameters | ~17M |
| Vocabulary | 8 tokens (G, A, C, T + special tokens) |
| Training data | Read-length DNA sequences derived from uniformly sampled genomes from across the prokaryotic tree of life |

## LookingGlass vocabulary

The vocabulary consists of 8 tokens:

| Token | ID | Description |
|-------|-----|-------------|
| `xxunk` | 0 | Unknown |
| `xxpad` | 1 | Padding |
| `xxbos` | 2 | Beginning of sequence |
| `xxeos` | 3 | End of sequence |
| `G` | 4 | Guanine |
| `A` | 5 | Adenine |
| `C` | 6 | Cytosine |
| `T` | 7 | Thymine |

# Installation

```bash
pip install torch
git clone https://huggingface.co/HoarfrostLab/lookingglass-v1
cd lookingglass-v1
```

# Tutorial

## Quick Start

```python
from lookingglass import LookingGlass, LookingGlassTokenizer

model = LookingGlass.from_pretrained('.')
tokenizer = LookingGlassTokenizer()

inputs = tokenizer(["GATTACA", "ATCGATCGATCG"], return_tensors=True)
embeddings = model.get_embeddings(inputs['input_ids'])
print(embeddings.shape)  # torch.Size([2, 104])
```

## Using Classifiers

```bash
pip install torch
git clone https://huggingface.co/HoarfrostLab/LGv1_OptimalTempClassifier
cd LGv1_OptimalTempClassifier
```

```python
from lookingglass_classifier import LookingGlassClassifier, LookingGlassTokenizer

model = LookingGlassClassifier.from_pretrained('.')
tokenizer = LookingGlassTokenizer()
model.eval()

inputs = tokenizer(["GATTACA"], return_tensors=True)
prediction = model.predict(inputs['input_ids'])
probs = model.predict_proba(inputs['input_ids'])

print(f"Predicted class: {model.config.class_names[prediction[0]]}")
print(f"Probabilities: {probs}")
```

## Getting Embeddings

The primary use case is extracting sequence embeddings for downstream tasks:

```python
from lookingglass import LookingGlass, LookingGlassTokenizer
import torch

model = LookingGlass.from_pretrained('./lookingglass-v1')
tokenizer = LookingGlassTokenizer()
model.eval()

# Your DNA sequences
sequences = [
    "ATCGATCGATCG",
    "GATTACAGATTACA",
    "GCGCGCGCGCGC"
]

# Tokenize
inputs = tokenizer(sequences, return_tensors=True)

# Extract embeddings
with torch.no_grad():
    embeddings = model.get_embeddings(inputs['input_ids'])

# embeddings: (3, 104) - one 104-dimensional vector per sequence
print(f"Embedding shape: {embeddings.shape}")
```

## Language Modeling

To access the full language model with prediction head:

```python
from lookingglass import LookingGlassLM, LookingGlassTokenizer

model = LookingGlassLM.from_pretrained('./lookingglass-v1')
tokenizer = LookingGlassTokenizer()

inputs = tokenizer("GATTACA", return_tensors=True)

# Get next-token prediction logits
logits = model(inputs['input_ids'])
print(logits.shape)  # torch.Size([1, 8, 8]) - (batch, seq_len, vocab_size)

# Embeddings also available
embeddings = model.get_embeddings(inputs['input_ids'])
```

## Batch Processing

```python
from lookingglass import LookingGlass, LookingGlassTokenizer
import torch

model = LookingGlass.from_pretrained('./lookingglass-v1')
tokenizer = LookingGlassTokenizer()
model.eval()

# Process sequences in batches
sequences = ["ATCG" * 25] * 100  # 100 sequences

batch_size = 32
all_embeddings = []

for i in range(0, len(sequences), batch_size):
    batch = sequences[i:i+batch_size]
    inputs = tokenizer(batch, return_tensors=True)

    with torch.no_grad():
        emb = model.get_embeddings(inputs['input_ids'])
    all_embeddings.append(emb)

embeddings = torch.cat(all_embeddings, dim=0)
print(f"Processed {len(embeddings)} sequences")
```

## GPU Usage

```python
import torch
from lookingglass import LookingGlass, LookingGlassTokenizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = LookingGlass.from_pretrained('./lookingglass-v1')
model = model.to(device)
model.eval()

tokenizer = LookingGlassTokenizer()

inputs = tokenizer(["GATTACA"], return_tensors=True)
input_ids = inputs['input_ids'].to(device)

with torch.no_grad():
    embeddings = model.get_embeddings(input_ids)
```

## Fine-tuning for Downstream Tasks

Add a classification head on top of LookingGlass embeddings for tasks like functional annotation:

```python
import torch
import torch.nn as nn
from lookingglass import LookingGlass, LookingGlassTokenizer

class SequenceClassifier(nn.Module):
    def __init__(self, num_classes, model_path='./lookingglass-v1', freeze_encoder=True):
        super().__init__()
        self.encoder = LookingGlass.from_pretrained(model_path)
        self.classifier = nn.Sequential(
            nn.Linear(104, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, input_ids):
        embeddings = self.encoder.get_embeddings(input_ids)
        return self.classifier(embeddings)

# Example: 3-class classifier (e.g., psychrophilic/mesophilic/thermophilic)
model = SequenceClassifier(num_classes=3)
tokenizer = LookingGlassTokenizer()

# Training loop
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

sequences = ["ATCGATCG", "GCGCGCGC", "TATATATA"]
labels = torch.tensor([0, 1, 2])

inputs = tokenizer(sequences, return_tensors=True)
logits = model(inputs['input_ids'])
loss = criterion(logits, labels)
loss.backward()
optimizer.step()
```

For gradual unfreezing (recommended for better performance):

```python
# Phase 1: Train classifier with frozen encoder
model = SequenceClassifier(num_classes=3, freeze_encoder=True)
tokenizer = LookingGlassTokenizer()
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(5):
    for batch_seqs, batch_labels in train_dataloader:
        optimizer.zero_grad()
        inputs = tokenizer(batch_seqs, return_tensors=True)
        logits = model(inputs['input_ids'])
        loss = criterion(logits, batch_labels)
        loss.backward()
        optimizer.step()

# Phase 2: Unfreeze encoder and fine-tune with lower learning rate
for param in model.encoder.parameters():
    param.requires_grad = True

optimizer = torch.optim.Adam([
    {'params': model.encoder.parameters(), 'lr': 1e-5},
    {'params': model.classifier.parameters(), 'lr': 1e-4}
])

for epoch in range(10):
    for batch_seqs, batch_labels in train_dataloader:
        optimizer.zero_grad()
        inputs = tokenizer(batch_seqs, return_tensors=True)
        logits = model(inputs['input_ids'])
        loss = criterion(logits, batch_labels)
        loss.backward()
        optimizer.step()
```

## Working with FASTA Files

Process sequences from FASTA files using BioPython (`pip install biopython`):

```python
from Bio import SeqIO
from lookingglass import LookingGlass, LookingGlassTokenizer
import torch

model = LookingGlass.from_pretrained('./lookingglass-v1')
tokenizer = LookingGlassTokenizer()
model.eval()

# Read all sequences from FASTA file
records = list(SeqIO.parse('sequences.fasta', 'fasta'))
headers = [record.id for record in records]
sequences = [str(record.seq).upper() for record in records]

# Get embeddings in batches
batch_size = 32
all_embeddings = []

for i in range(0, len(sequences), batch_size):
    batch = sequences[i:i+batch_size]
    inputs = tokenizer(batch, return_tensors=True)

    with torch.no_grad():
        emb = model.get_embeddings(inputs['input_ids'])
    all_embeddings.append(emb)

embeddings = torch.cat(all_embeddings, dim=0)

# Save embeddings with headers
import numpy as np
np.savez('embeddings.npz',
         embeddings=embeddings.numpy(),
         headers=headers)
```

For large FASTA files, use SeqIO.parse as a generator to avoid loading everything into memory:

```python
from Bio import SeqIO
from lookingglass import LookingGlass, LookingGlassTokenizer
import torch

model = LookingGlass.from_pretrained('./lookingglass-v1')
tokenizer = LookingGlassTokenizer()
model.eval()

def batch_records(filepath, batch_size=32):
    """Yield batches of records from a FASTA file."""
    batch = []
    for record in SeqIO.parse(filepath, 'fasta'):
        batch.append(record)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

# Process large file
for records in batch_records('large_file.fasta', batch_size=32):
    headers = [record.id for record in records]
    sequences = [str(record.seq).upper() for record in records]

    inputs = tokenizer(sequences, return_tensors=True)
    with torch.no_grad():
        embeddings = model.get_embeddings(inputs['input_ids'])
    # Process or save embeddings...
```

# API Reference

## LookingGlassTokenizer

```python
tokenizer = LookingGlassTokenizer(
    add_bos_token=True,   # Add xxbos at start (default: True)
    add_eos_token=False,  # Add xxeos at end (default: False)
)

# Tokenize
inputs = tokenizer(
    sequences,            # str or List[str]
    return_tensors=True,  # Return PyTorch tensors
    padding=True,         # Pad to longest sequence
    max_length=None,      # Optional max length
    truncation=False,     # Truncate to max_length
)

# Decode
tokenizer.decode(token_ids, skip_special_tokens=True)
```

## LookingGlass

```python
model = LookingGlass.from_pretrained(path)

# Get sequence embeddings (recommended)
embeddings = model.get_embeddings(input_ids)  # (batch, 104)

# Get hidden states for all positions
hidden = model.get_hidden_states(input_ids)   # (batch, seq_len, 104)

# Forward pass (same as get_embeddings)
embeddings = model(input_ids)                 # (batch, 104)
```

## LookingGlassLM

```python
model = LookingGlassLM.from_pretrained(path)

# Get logits for next-token prediction
logits = model(input_ids)                     # (batch, seq_len, 8)

# Get embeddings
embeddings = model.get_embeddings(input_ids)  # (batch, 104)
```

# Previous Releases

* [Release v1.0](https://github.com/ahoarfrost/LookingGlass/releases/tag/v1.0) - Original fastai-based models (requires fastai v1 and [fastBio](https://github.com/ahoarfrost/fastBio/))
