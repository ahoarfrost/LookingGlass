# Welcome to LookingGlass 

LookingGlass is a general-purpose 'universal language of life' deep learning model for read-length biological sequences. It can be used for diverse downstream transfer learning tasks for biological data, some of which are described in the paper.

This is the main repository for these pretrained models. Static URLs for downloading these models are available in [release v1](https://github.com/ahoarfrost/LookingGlass/releases/tag/v1.0) of this repo. See more detailed descriptions of the models contained in this repository below.

A python package - [fastBio](https://github.com/ahoarfrost/fastBio/) - for loading, manipulating, and training biological data for deep learning, as well as using the pretrained models in this release, is also available.

If you find LookingGlass, LookingGlass-derived models, or fastBio helpful, please cite the preprint:

> Hoarfrost, A., Aptekmann, A., Farfanuk, G. & Bromberg, Y. Shedding Light on Microbial Dark Matter with A Universal Language of Life. *bioRxiv* (2020). doi:10.1101/2020.12.23.424215. https://www.biorxiv.org/content/10.1101/2020.12.23.424215v2.

# Models in the most recent release

The most recent [release of LookingGlass](https://github.com/ahoarfrost/LookingGlass/releases/tag/v1.0) provides static URLs for available pretrained models.

* **LookingGlass**
    
    LookingGlass is a 'universal language of life', producing contextually-aware, functionally and evolutionarily relevant representations of short DNA reads. As a general purpose 'biological language' representation model, it is broadly useful for training diverse downstream transfer learning tasks. A number of exports are available:
    * [LookingGlass.pth](https://github.com/ahoarfrost/LookingGlass/releases/download/v1.0/LookingGlass.pth) - the full LookingGlass language model pretrained weights (encoder and decoder).
    * [LookingGlass_enc.pth](https://github.com/ahoarfrost/LookingGlass/releases/download/v1.0/LookingGlass_enc.pth) - the LookingGlass encoder weights only.
    * [LookingGlass_export.pkl](https://github.com/ahoarfrost/LookingGlass/releases/download/v1.0/LookingGlass_export.pkl) - the LookingGlass model as a fastai learner export (with empty databunch attached).

* **Functional Classifier**

    The functional classifier can classify DNA reads into one of 1247 experimentally-validated functional annotations with 81.5% accuracy. The following exports are available:
    * [FunctionalClassifier.pth](https://github.com/ahoarfrost/LookingGlass/releases/download/v1.0/FunctionalClassifier.pth) - the full functional classifier pretrained weights (encoder and decoder).
    * [FunctionalClassifier_enc.pth](https://github.com/ahoarfrost/LookingGlass/releases/download/v1.0/FunctionalClassifier_enc.pth) - the functional classifier encoder weights only.
    * [FunctionalClassifier_export.pkl](https://github.com/ahoarfrost/LookingGlass/releases/download/v1.0/FunctionalClassifier_export.pkl) - the functional classifier as a fastai learner export (with empty databunch attached).

* **Oxidoreductase Classifier**

    The oxidoreductase can classify whether a DNA read originates from an oxidoreductase (EC number 1.-.-.-) with 82.3% accuracy. The following exports are available:
    * [OxidoreductaseClassifier.pth](https://github.com/ahoarfrost/LookingGlass/releases/download/v1.0/OxidoreductaseClassifier.pth) - the full oxidoreductase classifier pretrained weights (encoder and decoder).
    * [OxidoreductaseClassifier_export.pkl](https://github.com/ahoarfrost/LookingGlass/releases/download/v1.0/OxidoreductaseClassifier_export.pkl) - the oxidoreductase as a fastai learner export. 

* **Optimal Temp Classifier**

    The optimal temperature classifier can identify whether a DNA read originates from an enzyme with a psychrophilic (<15 C), mesophilic (20-40 C), or thermophilic (>50 C) optimal temperature with 70.1% accuracy. The following exports are available:
    * [OptimalTempClassifier.pth](https://github.com/ahoarfrost/LookingGlass/releases/download/v1.0/OptimalTempClassifier.pth) - the full optimal temperature classifier pretrained weights (encoder and decoder). 

* **Reading Frame Classifier**

    The reading frame classifier identifies the correct reading frame start position (1, 2, 3, -1, -2, or -3) for DNA reads (and thus the true amino acid sequence). Note it is currently only intended for prokaryotic sequences with low proportions of noncoding DNA. The following exports are available:
    * [ReadingFrameClassifier.pth](https://github.com/ahoarfrost/LookingGlass/releases/download/v1.0/ReadingFrameClassifier.pth)

## LookingGlass vocabulary

The vocabulary used for training LookingGlass and all LookingGlass-derived models above is also available for direct download:

* [ngs_vocab_k1_withspecial.npy ](https://github.com/ahoarfrost/LookingGlass/releases/download/v1.0/ngs_vocab_k1_withspecial.npy) - maintains the vocabulary token order for correct numericalization of DNA sequences for LookingGlass-derived models.

# Tutorial

The [fastBio](https://github.com/ahoarfrost/fastBio/) python package can be used to directly create LookingGlass and LookingGlass-derived models (with or without pretrained weights). See the [fastBio tutorial](https://github.com/ahoarfrost/fastBio/blob/master/Tutorial.ipynb).

Alternatively, models described above were saved with pytorch.save() and can be loaded directly into your own scripts with pytorch.load().