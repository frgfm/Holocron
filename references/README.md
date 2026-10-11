# Holocron training scripts

## Installation

Use Python 3.11 or higher, [Git](https://git-scm.com/), and [uv](https://docs.astral.sh/uv/getting-started/installation/) to install the training dependencies:

```shell
git clone https://github.com/frgfm/Holocron.git
cd Holocron
uv sync --locked --extra training
```

Shared CLI setup and training actions live in [`_common.py`](_common.py). They belong to the reference recipes; the installed `holocron.trainer` package provides the training primitives.

## CPU, CUDA, and Apple Silicon

Classification, segmentation, detection, and character-training recipes accept
`--device cpu`, `--device cuda:0`, or `--device mps`. The shared recipes default to
`--device auto`, which chooses CUDA, then MPS, then CPU. The older `--device 0`
CUDA spelling still works. An unavailable explicitly requested accelerator raises
an error instead of silently changing devices.

```shell
uv run --no-sync python -m references.classification.train DATA --arch repvit_m0_9 --device cpu -j 4
uv run --no-sync python -m references.classification.train DATA --arch repvit_m0_9 --device cuda:0 --amp -j 4
uv run --no-sync python -m references.classification.train DATA --arch repvit_m0_9 --device mps --amp -j 4
```

CPU training uses FP32 by default; `--amp` opts into BF16 autocasting. CUDA and
MPS use FP16 autocasting with gradient scaling when `--amp` is enabled. Measure
CPU mixed precision on the target processor: it is not faster on every CPU.
Pinned host memory is used only for CUDA. For Imagenette on MPS, start with one
or two loading workers (`--workers 1` or `--workers 2`), or `-j 0` when debugging.
The Mixup loader supports macOS worker spawning. Local M3 Pro diagnostics found
that `--no-amp` reduced training time for RepViT and MobileOne-S2, but RepViT used
more Metal driver memory. Measure both modes for your model; these short runs do
not establish accuracy parity or optimal settings for other hardware or tasks.

For line recognition, use `--device mps` with the recognition script. Its CTC
loss runs on CPU because PyTorch has no MPS CTC kernel; gradients return to the
MPS model. Other training and inference remain on the requested device.

Add `--benchmark` to the standard classification training command to record
metrics and deployment profiling beside its checkpoint. There is no separate
benchmark training recipe. Benchmark mode requires fresh training, without
resume or diagnostic actions; normal W&B logging remains available.

The [RepViT campaign](classification/benchmark_repvit_imagenette.sh) runs that
same command for all three RepViT variants and MobileOne-S2:

```shell
bash references/classification/benchmark_repvit_imagenette.sh \
  /path/to/imagenette2-320 /path/to/fresh-results --device mps
```

The default is 20 epochs, seed 0, AMP, four workers, and batch size 32. For the MPS
settings above, append `--workers 1 --no-amp`; this changes the recipe, not the
historical benchmark results. To use a physical batch
of 8 with the same effective batch, pass `--batch-size 8 --grad-acc 4`. Other
options come directly from the training CLI. Each model runs in a fresh process.
Results include selected and final
validation accuracy, model size, convolution/linear MACs, synchronized deployment
latency, elapsed wall time, and backend-specific memory readings. MPS memory is
sampled tensor/driver allocation, not CUDA VRAM or an exact peak. Checkpoints and
progress records stay in the requested output directory, which must be empty
for the campaign. Each checkpoint has its own JSON report and progress file.

## Available tasks

### Image classification

Refer to the [`./classification`](classification) folder

### Semantic segmentation

Refer to the [`./segmentation`](segmentation) folder

### Object detection

Refer to the [`./detection`](detection) folder

### Character and text recognition

Refer to the [`./recognition`](recognition) folder for a reproducible character-pretraining, CTC line-recognition, and synthetic-page OCR experiment.
