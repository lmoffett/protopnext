# ProtoPNext

**This is a prerelease codebase. All APIs, interfaces, and implementations are subject to change without notice.**

ProtoPNext is a research library for prototypical part networks (PPNs) that provides implementations of various prototype-based interpretable models including ProtoPNet, TesNet, Deformable ProtoPNet, ProtoTree, and ST-ProtoPNet.

## Installing

`protopnext` currently supports python 3.8.
We recommend creating a dedicated python environment for development (i.e., with `virtualenv` or `conda`).

### Install Dependencies

You need to choose between CPU and CUDA-based development.
If you have an NVidia GPU, you should install the CUDA torch dependencies so you can switch between CPU and GPU execution.
Otherwise, use CPU.

For CPU:

```{sh}
pip install -r env/requirements-frozen.txt --extra-index-url=https://download.pytorch.org/whl/cpu
```

For CUDA:

```{sh}
pip install -r env/requirements-frozen.txt --extra-index-url=https://download.pytorch.org/whl/cu117
```

## Training Your First ProtoPNet

This will walk you through the commands to setup a CUB-200 dataset and train ProtoPNet.

### CUB-200 Dataset Prep

1. Download the dataset CUB_200_2011.tgz from https://www.vision.caltech.edu/datasets/cub_200_2011/ using the command `wget https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz`
2. Unpack CUB_200_2011.tgz. 
Check to see if the file unpacked properly
3. Run `python -m protopnet datasets create-splits /path/to/unpacked/CUB_200_2011`. This will create `train`, `validation`, and `test` directories in your `CUB_200_2011` directory that match the splits used for ProtoPNeXt.

### Training Your first ProtoPNet

On the isolated environment you set up with CUDA available, run the following:

1. Log in to the weights and biases with `wandb login`
2. Set your CUB_200 directory to the variable `export CUB200_DIR=/path/to/unpacked/CUB_200_2011`
3. Run `python -m protopnet train-protopnet`. See `protopnet.train_protopnet.main` for the default parameters. (Suggestion: It might be worth to change the `phase_multiplier` parameter value to something small like 1-5 for your first run of ProtoPNet to reduce the number of epochs in each training phase, but this is not necessary).

Local logging shows the progress of the training in Epochs.
In Weights and Biases, you can view the training curves.

### Visualizing Model Reasoning

There are several options for visualizing model reasoning. Start by training a model, for example:

`protopnet train-protopnet --dataset=shapes_2d --proto-channels=32 --num-prototypes-per-class=5 --output-dir=analysis`

Once trained, use the `viz` argument to render the prototypes:

`protopnet viz render-prototypes --model-path="/get/from/training/logs" --dataset=shapes_2d --output-dir=analyses`

This will produce multiple visualization for each prototype including:

- Original prototype image
- Heatmap showing activation
- Overlaid heatmap on the original image
- Original image with bounding box around the high-activation region
- In cases where parts are specified in the dataset, vizualization of part locations.

#### Local Analysis

It is also possible to do `local` analyses, focusing on a specific test instances and associated classifications.

`protopnet viz local --model-path="/get/from/training/logs" --dataset=shapes_2d --sample=<num_samples> --output-dir=analyses`

This will create a set of vizualizations including:

- Test image alongside activation prototypes
- Top-k prototypes that influenced the classification
- Similarity scores and connections to classes
- Contribution of each prototype to the final classification.

The `--sample` parameter is used to limit the number of test images that will be analyzed.

#### Global Analysis

Finally, `global` analyses are possible that examine prototypes behavior across the entire dataset. This will find the top N similar samples to the generated prototypes. It also provides heatmaps and overlays.

`protopnet viz global --model-path="/get/from/training/logs" --dataset=shapes_2d --sample=<num_prototypes> --output-dir=analysis`

If you call these without first having rendered prototypes, they will be automatically rendered for you.

## Code Organization

```
protopnet/
├── models/              # Model implementations
│   ├── vanilla_protopnet.py
│   ├── deformable_protopnet.py
│   └── ...
├── datasets/            # Dataset loaders and utilities
│   ├── torch_extensions.py
│   ├── cars_cropped.py
│   └── ...
├── train/              # Auto-training infrastructure
│   ├── scheduling/     # Training phase scheduling
│   ├── logging/        # Metrics and W&B integration
│   └── checkpointing.py
├── utilities/          # General utilities
│   ├── trainer_utilities.py
│   ├── visualization_utilities.py
│   └── ...
├── pretrained/         # Pretrained backbone implementations
├── activations.py      # Prototype activation functions
├── backbones.py        # Backbone model construction
├── embedding.py        # Feature embedding layers
├── prediction_heads.py # Classification heads
├── prototype_layers.py # Prototype computation layers
└── cli.py              # Command-line interface
```

### Core Components vs Training Infrastructure

The codebase is organized into two main categories:

**Core Library Components** are used for constructing models:

- `activations.py` - Cosine, L2, and other prototype activation functions
- `backbones.py` - Backbone model construction and configuration
- `embedding.py` - Feature embedding and addon layers  
- `prediction_heads.py` - Classification heads for prototype models
- `prototype_layers.py` - Core prototype computation and projection
- `models/` - Complete model implementations
- `datasets/` - Dataset loading and preprocessing
- `pretrained/` - Pretrained model implementations
- `utilities/` - General utilities for visualization, preprocessing, etc.

**Auto-Training Infrastructure** (`train/` package) is used for creating training loops:

- `scheduling/` - Multi-phase training schedules and early stopping
- `logging/` - Metrics tracking, W&B integration, and logging
- `checkpointing.py` - Model saving and loading utilities
- `types.py` - Training-related protocols and interfaces

## Key Concepts

### Prototype Models

All models inherit from `ProtoPNet` base class and consist of:

- **Backbone**: Feature extraction (ResNet, DenseNet, VGG, etc.)
- **Add-on layers**: Feature processing before prototype computation
- **Prototype layer**: Computes similarity between features and learned prototypes
- **Prediction head**: Maps prototype activations to class predictions

### Training Phases

Training alternates between multiple phases:

- **Warm-up**: Train only prototype layers (backbone frozen)
- **Joint**: End-to-end training of all components
- **Project**: Update prototypes to match training examples
- **Last-only**: Train only the classification head

### Early Stopping

Custom early stopping for prototype models:

- **Project patience**: Stop after N projections without improvement

## Command Line Interface

The library provides a CLI to train models based on command line configuration, which can be used as a starting point for experimenting with ProtoPNet models.
Custom models will require their own configurations.

```bash
# Train vanilla ProtoPNet
python -m protopnet train-protopnet --dataset cub200 --backbone resnet50

# Train deformable ProtoPNet  
python -m protopnet train-deformable --dataset cub200 --activation-function cosine

# Evaluate models
python -m protopnet eval models.csv --dataset cub200 --output results.csv

# Generate visualizations
python -m protopnet visualization model.pth --dataset cub200
```

## Datasets

Available datasets:

- **cub200**: CUB-200-2011 birds dataset
- **cub200_cropped**: Cropped version using bounding boxes
- **cars_cropped**: Stanford Cars with cropping
- **dogs**: Stanford Dogs dataset

## Library

### Training Phase System

Training alternates between *phases*, which are a series of *steps*.
`ClassificationBackpropPhase` is the standard gradient descent training phase, but configureswhich network layers to train.
Other phases have special non-gradient behaviors, like projection.

For example, the `ProtoPNetTrainingSchedule` has the following layout:

- **Warm-up**: `ClassificationBackpropPhase` training only prototype layers (and, optionally, last layer)
- **Joint**: `ClassificationBackpropPhase` training all layers
- **Project**: `ProjectPhase` - one epoch updating the prototypes to match the closest sample patches
- **Last-only**: `ClassificationBackpropPhase` training only the classification head

## Contributors

- Frank Willard
- Maximilian Machado
- Emanuel Mokel
- Adam Costarino
- Jon Donnelly
- Zhicheng Guo
- Dennis Tang
- Julia Yang
- Giyoung Kim
- Alina Jade Barnett

## References

1. Chen et al. This Looks Like That: Deep Learning for Interpretable Image Recognition. NeurIPS, 2019.
2. Donnelly et al. Deformable ProtoPNet: An Interpretable Image Classifier Using Deformable Prototypes. ICCV, 2022.
3. Nauta et al. Neural prototype trees for interpretable fine-grained image recognition. ICCV, 2021.
4. Wang et al. Interpretable Image Recognition by Constructing Transparent Embedding Space. ICCV, 2021.
5. Wang et al. Learning support and trivial prototypes for interpretable image classification. ICCV, 2023
6. Huang et al. Evaluation and improvement of interpretability for self-explainable part-prototype networks. ICCV, 2023.
7. Willard et al. This looks better than that: Better interpretable models with protopnext. arXiv preprint arXiv:2406.14675 (2024).
8. Moffett et al. Cosine Similarity is Almost All You Need (for Prototypical-Part Models). WACV, 2026.