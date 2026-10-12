# FMSP-Net: Training and Inference

FMSP-Net is a remote sensing semantic segmentation framework integrating Multi-Modal Spatial Anchoring (MSAM), Frequency Semantic Adapters (FSA), Multi-Scale Mamba Aggregation (MMAM), Language-Guided Semantic Activation (LSAM), and Dynamic Prototype Refinement (DPR).

This project provides model definitions, training, evaluation, and inference scripts. It reads prepared `.pt` tensors and does not include dataset preprocessing.

## 1. Environment

| Component | Configuration |
|---|---|
| GPU | NVIDIA A100-PCIE-40GB |
| CPU | Intel Xeon Skylake, 10 cores |
| RAM | 72 GB |
| OS | Linux x86_64 |
| Python | 3.10 |
| PyTorch | 2.0.1 |
| CUDA | 11.8 |
| SAM | Meta SAM ViT-H |
| CLIP | OpenCLIP ViT-B/32 |
| Mamba | `mamba-ssm` |

### Installation

```bash
conda create -n fmsp python=3.10 -y
conda activate fmsp
python -m pip install --upgrade pip setuptools wheel ninja packaging
python -m pip install torch==2.0.1 torchvision==0.15.2 \
  --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt --no-build-isolation
```

### Environment verification

```bash
nvidia-smi
nvcc --version
python -c "import torch, open_clip; from segment_anything import sam_model_registry; from mamba_ssm import Mamba; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

## 2. Project Structure

```text
FMSP-Net_A100_runnable/
├── README.md
├── README_ZH.md
├── INSTALL_A100.md
├── requirements.txt
├── requirements_A100_mamba_compat.txt
├── requirements-test.txt
├── configs/
│   ├── __init__.py
│   └── datasets.py
├── models/
│   ├── __init__.py
│   ├── backbone.py
│   ├── msam.py
│   ├── fsa.py
│   ├── mmam.py
│   ├── text.py
│   ├── lsam.py
│   ├── dpr.py
│   ├── decoder.py
│   ├── losses.py
│   └── fmsp_net.py
├── checkpointing.py
├── data.py
├── metrics.py
├── train.py
├── evaluate.py
├── inference.py
├── verify_backend.py
├── tests/
│   └── test_model.py
├── examples/
└── checkpoints/
```

## 3. Model Components

| Module | Source | Function |
|---|---|---|
| SAM ViT-H | `models/backbone.py` | Extract hierarchical features from layers 6, 12, 18, and 24 |
| MSAM | `models/msam.py` | Fuse optical and DSM representations |
| FSA | `models/fsa.py` | Frequency separation using FFT/IFFT and learnable masks |
| MMAM | `models/mmam.py` | Aggregate multi-scale representations using bidirectional Mamba |
| CLIP / LSAM | `models/text.py`, `models/lsam.py` | Construct class prototypes and category activation maps |
| DPR | `models/dpr.py` | Update prototype memory and refine visual features |
| Decoder | `models/decoder.py` | Generate semantic predictions from visual and prompt features |
| Joint Loss | `models/losses.py` | Cross-entropy, Dice, boundary and prototype losses |

## 4. Input Format

Each dataset sample is stored in a `.pt` file containing a Python dictionary.

| Key | Tensor shape | Data type | Requirement |
|---|---|---|---|
| `image` | `[3, H, W]` | `torch.float32`, values in `[0, 1]` | Required |
| `mask` | `[H, W]` | `torch.int64`, labels `0...K-1`, ignore label `255` | Training/evaluation |
| `dsm` | `[1, H, W]` | `torch.float32`, values in `[0, 1]` | Potsdam/Vaihingen |

Training and evaluation use a manifest containing a JSON list of sample paths:

```json
["sample_0001.pt", "sample_0002.pt"]
```

Input images and DSM tensors must already be spatially aligned. LoveDA and UAVid do not require DSM inputs.

## 5. Unit Tests and CPU Smoke Test

```bash
python -m pip install -r requirements-test.txt
python -m unittest discover -s tests -v
python train.py --smoke-test --dataset potsdam --epochs 1 \
  --smoke-size 32 --save-dir checkpoints/potsdam_smoke
python inference.py \
  --checkpoint checkpoints/potsdam_smoke/best.pt \
  --input examples/cpu_input.pt \
  --output examples/potsdam_pred.pt \
  --device cpu --allow-smoke-checkpoint
python evaluate.py \
  --checkpoint checkpoints/potsdam_smoke/best.pt \
  --manifest examples/cpu_example_manifest.json \
  --device cpu --allow-smoke-checkpoint
```

## 6. SAM, CLIP, and Mamba Integration Test

Provide an official SAM ViT-H checkpoint and install the corresponding CLIP and Mamba dependencies.

```bash
python verify_backend.py \
  --sam-checkpoint /path/to/sam_vit_h_4b8939.pth \
  --dataset potsdam --patch-size 512 --device cuda
```

## 7. Training

```bash
python train.py \
  --dataset potsdam \
  --train-manifest /data/potsdam/train.json \
  --val-manifest /data/potsdam/val.json \
  --sam-checkpoint /weights/sam_vit_h_4b8939.pth \
  --backbone sam --ssm mamba --text open_clip \
  --sam-input-size 1024 \
  --feature-dim 256 --dsm-dim 64 --d-state 16 \
  --batch-size 1 --grad-accum-steps 4 \
  --epochs 300 --lr 0.0002 \
  --device cuda --save-dir checkpoints/potsdam_run
```

### Resume training

```bash
python train.py \
  --dataset potsdam \
  --train-manifest /data/potsdam/train.json \
  --val-manifest /data/potsdam/val.json \
  --sam-checkpoint /weights/sam_vit_h_4b8939.pth \
  --resume checkpoints/potsdam_run/last.pt \
  --device cuda --save-dir checkpoints/potsdam_run
```

## 8. Inference

```bash
python inference.py \
  --checkpoint checkpoints/potsdam_run/best.pt \
  --sam-checkpoint /weights/sam_vit_h_4b8939.pth \
  --input /data/potsdam/test_0001.pt \
  --output examples/test_0001_prediction.pt \
  --device cuda
```

The saved result contains the following tensors:

| Name | Shape | Description |
|---|---|---|
| `prediction` | `[1, H, W]` | Predicted integer segmentation mask |
| `probabilities` | `[1, K, H, W]` | Per-class softmax probabilities |
| `activation` | `[1, K, h, w]` | Text-guided activation maps |

## 9. Evaluation

```bash
python evaluate.py \
  --checkpoint checkpoints/potsdam_run/best.pt \
  --sam-checkpoint /weights/sam_vit_h_4b8939.pth \
  --manifest /data/potsdam/test.json \
  --device cuda
```

To train on another dataset, set `--dataset` to `vaihingen`, `loveda`, or `uavid` and use the matching data and checkpoint.

## 10. Reproducibility Status

This is an independently reconstructed implementation based on the FMSP-Net manuscript rather than the authors' original verified research code. The decoder wiring, prompt wording, and some architectural details are implementation choices where the manuscript does not specify exact executable configurations. The included CPU smoke checkpoint was trained on synthetic samples and does not reproduce the reported benchmark results. Paper-level reproduction requires the original experimental configuration, prepared datasets, and research-trained model weights.

To export the installed dependencies after validating the A100 environment:

```bash
python -m pip freeze > requirements-locked-a100.txt
```
