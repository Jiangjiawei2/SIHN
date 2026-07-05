# SIHN: Semantic-Injected Hierarchical Network

This repository contains the PyTorch implementation of **SIHN: A
Semantic-Injected Hierarchical Network for Accelerated MRI Reconstruction**.
SIHN combines a model-driven data-consistency path with hierarchical semantic
feature modeling for single-coil and multi-coil MRI reconstruction.

## Highlights

- ADMM-inspired unfolding with explicit data-consistency updates.
- Contextual Extraction Unit (CEU) and Interscale Aggregation Unit (IAU) for
  hierarchical semantic feature flow.
- Semantic Graph Reasoning Module (SGRM) for latent structural relationship
  modeling.
- Dual-Scale Attention Module (DSAM) for local detail refinement.
- PyTorch Lightning training, validation, testing, checkpointing, and logging.
- Configurations for fastMRI, IXI, and MoDL-style multi-coil data.

## Repository Layout

```text
.
|-- configs/                 # Experiment configurations
|-- scripts/                 # Convenience training scripts
|-- sihn/
|   |-- data/                # HDF5 MRI datasets and datamodule
|   |-- models/              # SIHN modules, losses, and MRI operators
|   `-- utils/               # FFT, masks, metrics, and seeding helpers
|-- tests/                   # Config, mask, and model-shape smoke tests
|-- train.py                 # Training entry point
|-- evaluate.py              # Checkpoint evaluation entry point
`-- infer.py                 # Single-volume inference entry point
```

## Installation

```bash
conda create -n sihn python=3.10 -y
conda activate sihn
pip install -r requirements.txt
```

For editable development:

```bash
pip install -e .
```

## Data

This repository does not redistribute MRI datasets. Download each dataset from
its official source and keep the original files unchanged.

- fastMRI: https://fastmri.med.nyu.edu/
- IXI: https://brain-development.org/ixi-dataset/
- MoDL multi-coil example data: https://github.com/hkaggarwal/modl

The loaders expect HDF5 files organized by split:

```text
data/
|-- fastmri/
|   |-- train/*.h5
|   |-- val/*.h5
|   `-- test/*.h5
|-- ixi/
|   |-- train/*.h5
|   |-- val/*.h5
|   `-- test/*.h5
`-- modl_multicoil/
    |-- train/*.h5
    |-- val/*.h5
    `-- test/*.h5
```

Update `data.root`, split names, mask settings, and output directories in the
YAML files under `configs/` as needed.

## Training

Single-coil fastMRI:

```bash
python train.py --config configs/fastmri_singlecoil.yaml
```

Single-coil IXI:

```bash
python train.py --config configs/ixi_singlecoil.yaml
```

Multi-coil MoDL-style data:

```bash
python train.py --config configs/modl_multicoil.yaml
```

Convenience scripts are also provided:

```bash
bash scripts/train_fastmri_a100.sh
bash scripts/train_ixi_a100.sh
bash scripts/train_modl_multicoil_a100.sh
```

## Evaluation

```bash
python evaluate.py \
  --config configs/fastmri_singlecoil.yaml \
  --checkpoint outputs/fastmri_sihn/checkpoints/last.ckpt
```

Metrics are selected through `eval.metrics` in each config. Supported metrics
include `psnr`, `ssim`, `nrmse`, and `lpips`.

## Inference

```bash
python infer.py \
  --config configs/fastmri_singlecoil.yaml \
  --checkpoint outputs/fastmri_sihn/checkpoints/last.ckpt \
  --input-npy path/to/complex_kspace.npy \
  --output-npy outputs/reconstruction.npy
```

The inference input is a complex-valued NumPy k-space array. Lightning
checkpoints and raw SIHN state dictionaries are both supported.

## Reproducibility Defaults

- Random seed: `2024`
- Unfolding stages: `5`
- Optimizer: Adam, betas `(0.9, 0.999)`
- Initial learning rate: `1e-4`
- Epochs: `300`
- Loss: Charbonnier
- Framework: PyTorch Lightning
- Masks: random Cartesian, equispaced Cartesian, and radial
- Supported settings: single-coil and multi-coil complex k-space

Representative trained checkpoints will be released with the public repository
after publication.

## Tests

Config and mask tests run without PyTorch. Model-shape tests require PyTorch.

```bash
PYTHONPATH=. python -m unittest discover -s tests -v
```

## Citation

If this code is useful for your research, please cite the SIHN paper. The
citation entry will be updated after publication.

```bibtex
@article{hong2026sihn,
  title  = {SIHN: A Semantic-Injected Hierarchical Network for Accelerated MRI Reconstruction},
  author = {Hong, Yupeng and Jiang, Jiawei and Quan, Yueqian and Jing, Liang and He, Yuxin and Zheng, Jianwei and Cai, Jing},
  year   = {2026}
}
```
