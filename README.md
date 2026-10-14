<p align="center">
  <img src="https://github.com/frgfm/Holocron/releases/download/v0.1.3/holocron_logo_text.png" width="40%">
</p>

<p align="center">
  <a href="https://github.com/frgfm/Holocron/actions/workflows/build.yml">
    <img alt="CI Status" src="https://img.shields.io/github/actions/workflow/status/frgfm/holocron/package.yml?branch=main&label=CI&logo=github&style=flat-square">
  </a>
  <a href="https://github.com/astral-sh/ruff">
    <img src="https://img.shields.io/badge/Linter-Ruff-FCC21B?style=flat-square&logo=ruff&logoColor=white" alt="ruff">
  </a>
  <a href="https://github.com/astral-sh/ty">
    <img src="https://img.shields.io/badge/Typecheck-Ty-261230?style=flat-square&logo=astral&logoColor=white" alt="ty">
  </a>
  <a href="https://www.codacy.com/gh/frgfm/Holocron/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=frgfm/Holocron&amp;utm_campaign=Badge_Grade"><img src="https://app.codacy.com/project/badge/Grade/49fc8908c44b45d3b64131e49558f1e9"/></a>
  <a href="https://codecov.io/gh/frgfm/holocron">
    <img src="https://img.shields.io/codecov/c/github/frgfm/holocron.svg?logo=codecov&style=flat-square" alt="Test coverage percentage">
  </a>
</p>
<p align="center">
  <a href="https://pypi.org/project/pylocron/">
    <img src="https://img.shields.io/pypi/v/pylocron.svg?logo=python&logoColor=fff&style=flat-square&label=PyPI" alt="PyPi Status">
  </a>
  <img alt="GitHub release (latest by date)" src="https://img.shields.io/github/v/release/frgfm/holocron?label=Release&logo=github">
  <img src="https://img.shields.io/pypi/pyversions/pylocron.svg?logo=Python&label=Python&logoColor=fff&style=flat-square" alt="pyversions">
  <a href="https://github.com/frgfm/Holocron/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/frgfm/Holocron.svg?label=License&logoColor=fff&style=flat-square" alt="License">
  </a>
</p>
<p align="center">
  <a href="https://huggingface.co/spaces/frgfm/holocron">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue" alt="Huggingface Spaces">
  </a>
  <a href="https://colab.research.google.com/github/frgfm/notebooks/blob/main/holocron/quicktour.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab">
  </a>
</p>
<p align="center">
  <a href="https://frgfm.github.io/holocron">
    <img src="https://img.shields.io/github/actions/workflow/status/frgfm/holocron/page-build.yml?branch=main&label=Documentation&logo=read-the-docs&logoColor=white&style=flat-square" alt="Documentation Status">
  </a>
</p>

Implementations of recent Deep Learning tricks in Computer Vision, easily paired up with your favorite framework and model zoo.

> **Holocrons** were information-storage [datacron](https://starwars.fandom.com/wiki/Datacron) devices used by both the [Jedi Order](https://starwars.fandom.com/wiki/Jedi_Order) and the [Sith](https://starwars.fandom.com/wiki/Sith) that contained ancient lessons or valuable information in [holographic](https://starwars.fandom.com/wiki/Hologram) form.

*Source: Wookieepedia*

## Quick Tour
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/frgfm/notebooks/blob/main/holocron/quicktour.ipynb)

Holocron provides research implementations with PyTorch interfaces.
Implementations and training recipes may differ from a paper author's code.
This example uses an explicit checkpoint and its preprocessing and labels:

<!-- quickstart-example-start -->
```python
import torch
from PIL import Image
from holocron.models.classification import ResNet18_Checkpoint, resnet18

checkpoint = ResNet18_Checkpoint.IMAGENETTE.value
model = resnet18(checkpoint=checkpoint).eval()

image = Image.open(path_to_an_image).convert("RGB")
input_tensor = checkpoint.transforms()(image).unsqueeze(0)

with torch.inference_mode():
    probabilities = model(input_tensor).squeeze(0).softmax(dim=0)

class_idx = probabilities.argmax().item()
label = checkpoint.meta.categories[class_idx]
confidence = probabilities[class_idx].item()
print(label, confidence)
```
<!-- quickstart-example-end -->


## Pretrained models

An architecture may be available without pretrained weights. See the
[checkpoint list and loading guide](https://frgfm.github.io/holocron/reference/models/models/#available-checkpoints)
for classification datasets, metrics and weight compatibility, and the
[support status](https://frgfm.github.io/holocron/reference/models/models/#support-status)
for other tasks. Most classification checkpoints target Imagenette (10 classes);
selected ReXNet variants also provide ImageNet-1K weights (1,000 classes).

Use the checkpoint's preprocessing and category metadata, as shown above.
A matching model name does not make torchvision or `timm` weights compatible.
To adapt pretrained weights to your own classes, load them before replacing the
classifier; see the [transfer-learning guide](https://frgfm.github.io/holocron/getting-started/classification/).

Legacy weight loaders log a warning and keep the affected model's initial
parameters when no weights are available. YOLO26 builders instead raise
`ValueError` for `pretrained=True`. No full detection checkpoints are published; a pretrained
classification backbone is not a pretrained detector. Train models without
weights with the [reference scripts](references/).

## Loss functions

`PolyLoss` expects raw logits and either `torch.int64` class indices or soft
class probabilities. See the [loss input guide and runnable example](https://frgfm.github.io/holocron/reference/nn/#loss-functions)
for tensor shapes, target types and the current `ignore_index` limitations.


## Installation

### Prerequisites

Python 3.11 (or higher) and [uv](https://docs.astral.sh/uv/)/[pip](https://pip.pypa.io/en/stable/) are required to install Holocron.

### Latest stable release

You can install the last stable release of the package using [pypi](https://pypi.org/project/pylocron/) as follows:

```shell
pip install pylocron
```

### Developer mode

Alternatively, if you wish to use the latest features of the project that haven't made their way to a release yet, you can install the package from source *(install [Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) first)*:

```shell
git clone https://github.com/frgfm/Holocron.git
pip install -e Holocron/.
```

## Paper references

### PyTorch layers for every need
- Activation: [HardMish](https://github.com/digantamisra98/H-Mish), [NLReLU](https://arxiv.org/abs/1908.03682), [FReLU](https://arxiv.org/abs/2007.11824)
- Loss: [Focal Loss](https://arxiv.org/abs/1708.02002), MultiLabelCrossEntropy, [MixupLoss](https://arxiv.org/pdf/1710.09412.pdf), [ClassBalancedWrapper](https://arxiv.org/abs/1901.05555), [ComplementCrossEntropy](https://arxiv.org/abs/2009.02189), [MutualChannelLoss](https://arxiv.org/abs/2002.04264), [DiceLoss](https://arxiv.org/abs/1606.04797), [PolyLoss](https://arxiv.org/abs/2204.12511)
- Convolutions: [NormConv2d](https://arxiv.org/pdf/2005.05274v2.pdf), [Add2d](https://arxiv.org/pdf/1912.13200.pdf), [SlimConv2d](https://arxiv.org/pdf/2003.07469.pdf), [PyConv2d](https://arxiv.org/abs/2006.11538), [Involution](https://arxiv.org/abs/2103.06255)
- Regularization: [DropBlock](https://arxiv.org/abs/1810.12890)
- Pooling: [BlurPool2d](https://arxiv.org/abs/1904.11486), [SPP](https://arxiv.org/abs/1406.4729), [ZPool](https://arxiv.org/abs/2010.03045)
- Attention: [SAM](https://arxiv.org/abs/1807.06521), [LambdaLayer](https://openreview.net/forum?id=xTJEN-ggl1b), [TripletAttention](https://arxiv.org/abs/2010.03045)

### Models for vision tasks
- Image Classification: [Res2Net](https://arxiv.org/abs/1904.01169) (based on the great [implementation](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/res2net.py) from Ross Wightman), [Darknet-24](https://pjreddie.com/media/files/papers/yolo_1.pdf), [Darknet-19](https://pjreddie.com/media/files/papers/YOLO9000.pdf), [Darknet-53](https://pjreddie.com/media/files/papers/YOLOv3.pdf), [CSPDarknet-53](<https://arxiv.org/abs/1911.11929>), [ResNet](https://arxiv.org/abs/1512.03385), [ResNeXt](https://arxiv.org/abs/1611.05431), [TridentNet](https://arxiv.org/abs/1901.01892), [PyConvResNet](https://arxiv.org/abs/2006.11538), [ReXNet](https://arxiv.org/abs/2007.00992), [SKNet](https://arxiv.org/abs/1903.06586), [RepVGG](https://arxiv.org/abs/2101.03697), [ConvNeXt](https://arxiv.org/abs/2201.03545), [MobileOne](https://arxiv.org/abs/2206.04040).
- Object Detection: [YOLOv1](https://pjreddie.com/media/files/papers/yolo_1.pdf), [YOLOv2](https://pjreddie.com/media/files/papers/YOLO9000.pdf), [YOLOv4](https://arxiv.org/abs/2004.10934)
- Semantic Segmentation: [U-Net](https://arxiv.org/abs/1505.04597), [UNet++](https://arxiv.org/abs/1807.10165), [UNet3+](https://arxiv.org/abs/2004.08790)

### Vision-related operations
- boxes: [Distance-IoU & Complete-IoU losses](https://arxiv.org/abs/1911.08287)

### Trying something else than Adam
- Optimizer: [LARS](https://arxiv.org/abs/1708.03888), [Lamb](https://arxiv.org/abs/1904.00962), [TAdam](https://arxiv.org/abs/2003.00179), [AdamP](https://arxiv.org/abs/2006.08217), [AdaBelief](https://arxiv.org/abs/2010.07468), [Adan](https://arxiv.org/abs/2208.06677), and customized versions (RaLars), [AdEMAMix](https://arxiv.org/abs/2409.03137)
- Optimizer wrapper: [Lookahead](https://arxiv.org/abs/1907.08610), Scout (experimental)



## More goodies

### Documentation

The full package documentation is available [here](https://frgfm.github.io/holocron/) for detailed specifications.

### Demo app

The project includes a minimal demo app using [Gradio](https://gradio.app/)

![demo_app](https://github.com/frgfm/Holocron/releases/download/v0.1.3/holocron_gradio.png)

You can check the live demo, hosted on :hugs: [HuggingFace Spaces](https://huggingface.co/spaces) :hugs: over here :point_down:
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/frgfm/Holocron)


### Reference scripts

Reference scripts are provided to train your models using holocron on famous public datasets. Those scripts currently support the following vision tasks:
- [Image classification](references/classification)
- [Object detection](references/detection)
- [Semantic segmentation](references/segmentation)

### YOLO training status

YOLOv4's training implementation has been repaired and verified with regression tests and a reproducible CPU fixed-batch learning check. With a frozen pretrained Imagenette CSPDarknet53-Mish backbone, it produces correctly labeled detections with IoUs of 0.81 and 0.79 on two synthetic examples, with finite losses and gradients and a saved checkpoint. See the [detection guide](references/detection/README.md) for the command, matched control, and CUDA/VOC smoke recipe.

Full CUDA/VOC training on the repaired implementation remains pending; the synthetic result does not establish dataset accuracy or reproduce the paper. YOLOv1/v2 loss behavior is unchanged, with regression coverage for shared inference changes. YOLOv3 is not implemented as a detector; the Darknet-53 classification backbone is available. No pretrained detection checkpoints are published.

### Latency benchmark

Compare every model on your own hardware with the [local inference measurements](https://frgfm.github.io/holocron/reference/models/models/#local-inference-measurements).

Measure a classification model on your hardware:

```bash
uv sync --locked --extra scripts
uv run --no-sync python scripts/eval_latency.py rexnet1_0x --device cpu --output /tmp/latency.json
```

The command measures PyTorch on CPU by default. Use `--device cuda:0` or `--device mps` to measure a GPU.
Use `--backend onnx` to measure ONNX Runtime on CPU. Its export runs in a separate process and uses temporary files.
The ONNX worker checks its output against the PyTorch export reference.

Each command runs five fresh worker processes by default. It reports the first forward call, median and 95th-percentile
warmed batch latency, and sustained throughput in images per second. GPU timing waits for work to finish.
The summary uses medians across workers for timings and throughput. It shows the range of worker medians.
Peak RSS is the highest resident process memory across workers, recorded after inference.
It includes imports, model setup, and warm-up. It excludes the parent process, ONNX export, and output checks.
RSS is unavailable on Windows.
CUDA workers also save peak tensor allocations and allocator reservations in the JSON file.
RSS and CUDA memory are separate measurements. First-call timing excludes imports and model setup.
Compare RSS only between runs using the same runtime and device.

Use `--batch-size`, `--size`, `--threads`, `--it`, `--warmup`, and `--repeat` to set the workload.
Inputs use float32. Inference uses evaluation mode, disables gradient tracking, and converts reparametrizable
models to their inference form. CPU operations use one thread by default.
The fixed seed improves repeatability; it does not prove identical weights or inputs across PyTorch versions.
These measurements cover prepared-tensor inference. They exclude image loading, preprocessing, and transfers.
They do not measure training or model accuracy.

To compare main with a dependency PR, install each checkout into a separate environment with `uv sync --locked`.
Use the same Python version, machine, runtime, device, and settings. Run the same version of this script against
both installations, using an absolute path to the script. Save a JSON file from each environment.
The files record the package revision and dirty state, dependency versions, runtime settings, script hash,
and individual trials. Check that the CUDA math settings match before comparing GPU results.
Alternate baseline and candidate commands, and first run main twice to check normal measurement variation.
Compare latency, throughput, and peak RSS per workload; avoid treating small changes within that variation as gains.
CI runs short CPU checks for PyTorch and ONNX and uploads their JSON results. CI timing is advisory.

*All arguments are listed by `python scripts/eval_latency.py --help`.*

### Docker container

If you wish to deploy containerized environments, you can use the provided Dockerfile to build a docker image:

```shell
make build-api
```

### Minimal API template

Looking for a boilerplate to deploy a model from Holocron with a REST API? Thanks to the wonderful [FastAPI](https://github.com/tiangolo/fastapi) framework, you can do this easily.

#### Deploy your API locally

Run your API in a docker container as follows:

```shell
make start-api
```

*In order to stop the container, use `make stop-api`*

#### What you have deployed

Your API is now running on port 8080, with its documentation http://localhost:8080/redoc and requestable routes:

```python
import requests

with open("/path/to/your/img.jpeg", "rb") as f:
    data = f.read()
response = requests.post("http://localhost:8080/classification", files={"file": data}).json()
```



## Citation

If you wish to cite this project, feel free to use this [BibTeX](http://www.bibtex.org/) reference:

```bibtex
@software{Fernandez_Holocron_2020,
author = {Fernandez, François-Guillaume},
month = {5},
title = {{Holocron}},
url = {https://github.com/frgfm/Holocron},
year = {2020}
}
```


## Contributing

Any sort of contribution is greatly appreciated!

You can find a short guide in [`CONTRIBUTING`](CONTRIBUTING.md) to help grow this project!



## License

Distributed under the Apache 2.0 License. See [`LICENSE`](LICENSE) for more information.
