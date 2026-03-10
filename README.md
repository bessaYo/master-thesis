# Efficient and Interpretable Dynamic Slicing of Neural Networks: A Block-Based Approach

## Prerequisites
 Make sure that Python (>=3.10) and pip are installed on your system. You can verify their installation by running:

 ```bash
 python --version
 pip --version
 ```

## Create and Activate a Virtual Environment
A virtual environment ensures that all project dependencies are installed in an isolated directory. 
In this way, version conflicts are prevented and the setup is fully reproducible.
You can create and activate a virtual environment in python with:

```bash
python -m venv venv
source venv/bin/activate     # for macOS/Linux
venv\Scripts\activate        # for Windows
```


## Project Setup
The project setup and dependencies are defined in `pyproject.toml`. Run the following command to install the project with all required dependencies:

```bash
pip install -e .
```

## Slicing

Once the project dependencies are installed, slicing can be run with `main.py`:

```bash
python main.py --model resnet_cifar --target 0
```

### Arguments

Several arguments exist and can be added into the slicing command to control model, slicing techniques and images to slice:

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--model` | yes | — | Model name: `resnet_cifar`, `resnet18`, `resnet34`, `resnet50`, `resnet101` |
| `--target` | yes | — | Target class index for slicing |
| `--theta` | no | `0.3` | Contribution threshold for neuron filtering |
| `--channel_alpha` | no | `None` | Channel slicing parameter |
| `--block_beta` | no | `None` | Block slicing parameter |
| `--image_index` | no | `0` | Index of the test image in the dataset |
| `--save` | no | `false` | Save results as JSON to `evaluation/slices/` |

### Examples

```bash
# Slice airplane (class 0) on CIFAR-10
python main.py --model resnet_cifar --target 0 --theta 0.3

# Slice for "dog" (class 5) with channel and block filtering on ImageNet
python main.py --model resnet18 --target 5 --channel_alpha 0.8 --block_beta 0.8

# Different image and target class
python main.py --model resnet34 --target 3 --image_index 10
```

## Repository Structure

The slicing pipeline with all three phases (profiling, forward analysis, backward analysis) is implemented in `core/`. `models/` contains the currently supported model architectures. `evaluation/` holds the scripts for the three research questions. `pretrained/` stores model weights and activation profiles that were computed on Google Colab (A100). `tests/` contains pytest unit tests to verify correctness of the implementation.

```
├── main.py                  # CLI entry point for slicing
├── pyproject.toml           # Dependencies & config
├── core/
│   ├── slicer.py            # Main pipeline (profiling -> forward analysis -> backward analysis)
│   ├── graph.py             # Graph wrapper for torch.fx
│   ├── block.py             # Block structure analysis for ResNets
│   ├── filtering.py         # Theta, channel and block slicing
│   └── tracing/
│       ├── profiler.py      # Phase 1: mean activations
│       ├── forward.py       # Phase 2: activation deltas
│       ├── backward.py      # Phase 3: contribution propagation
│       ├── operations.py    # Backward contribution operations for different layers
│       └── base.py          # Shared analyzer class
├── models/
│   ├── resnet.py            # Custom ResNet for CIFAR-10 dataset
│   ├── imagenet.py          # ResNets (18/34/50/101)
│   └── simple.py            # Simple test networks (PaperNN, SimpleCNN)
├── evaluation/
│   ├── rq1/                 # Slice size and runtime tables
│   ├── rq2/                 # Counterfactual and pruning
│   ├── rq3/                 # Contribution similarity
│   └── slices/              # Slice results (JSON)
├── pretrained/
│   ├── checkpoints/         # Pretrained model weights
│   ├── profiles/            # Precomputed profiling means
│   └── profile.py           # Script for profiling
├── utils/
│   ├── data.py              # Dataset loading for CIFAR-10 and ImageNet
│   ├── evaluation.py        # Slice computation & aggregation
│   ├── report.py            # Logging report for main.py
│   └── tensor.py            # Tensor utilities
└── tests/                   # Pytest unit tests
```

