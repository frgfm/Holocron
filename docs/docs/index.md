# Holocron: a Deep Learning toolbox for PyTorch

<p align="center">
    <img src="img/logo_text.svg" alt="Holocron logo" width="50%">
</p>

Holocron is meant to bridge the gap between [PyTorch](https://pytorch.org/) and latest research papers. It brings training components that are not available yet in PyTorch with a similar interface.

!!! info "Development documentation"
    These pages follow the `main` branch. The latest PyPI release may not include recent changes.

This project is meant for:

* :zap: **speed**: architectures in this repo are picked for both pure performances and minimal latency
* :woman_scientist: **research**: train your models easily to SOTA standards

## Installation

Create and activate a virtual environment and then install Holocron:

```shell
uv pip install pylocron
```

For development and system-wide options, see the
[installation guide](getting-started/installation.md).

## Quick start

Load a checkpoint and use its preprocessing and category metadata:

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

To adapt this checkpoint to your own classes, follow the
[classification and transfer-learning guide](getting-started/classification.md).

## Model zoo

Holocron implements all three tasks below, but they do not have the same level
of checkpoint and benchmark coverage. See the
[capability and maturity matrix](reference/models/models.md#support-status)
before choosing a model.

### Image classification — validated checkpoints

See the [available checkpoints](reference/models/models.md#available-checkpoints)
for datasets, metrics, weight compatibility and loading instructions. Most
classification checkpoints target Imagenette (10 classes); selected ReXNet
variants also provide ImageNet-1K weights (1,000 classes).

* TridentNet from ["Scale-Aware Trident Networks for Object Detection"](https://arxiv.org/pdf/1901.01892.pdf)
* SKNet from ["Selective Kernel Networks"](https://arxiv.org/pdf/1903.06586.pdf)
* PyConvResNet from ["Pyramidal Convolution: Rethinking Convolutional Neural Networks for Visual Recognition"](https://arxiv.org/pdf/2006.11538.pdf)
* ReXNet from ["ReXNet: Diminishing Representational Bottleneck on Convolutional Neural Network"](https://arxiv.org/pdf/2007.00992.pdf)
* RepVGG from ["RepVGG: Making VGG-style ConvNets Great Again"](https://arxiv.org/pdf/2101.03697.pdf)

### Semantic segmentation — unbenchmarked

* U-Net from ["U-Net: Convolutional Networks for Biomedical Image Segmentation"](https://arxiv.org/pdf/1505.04597.pdf)
* U-Net++ from ["UNet++: Redesigning Skip Connections to Exploit Multiscale Features in Image Segmentation"](https://arxiv.org/pdf/1912.05074.pdf)
* UNet3+ from ["UNet 3+: A Full-Scale Connected UNet For Medical Image Segmentation"](https://arxiv.org/pdf/2004.08790.pdf)

### Object detection — experimental

* YOLO from ["You Only Look Once: Unified, Real-Time Object Detection"](https://pjreddie.com/media/files/papers/yolo_1.pdf)
* YOLOv2 from ["YOLO9000: Better, Faster, Stronger"](https://pjreddie.com/media/files/papers/YOLO9000.pdf)
* YOLOv4 from ["YOLOv4: Optimal Speed and Accuracy of Object Detection"](https://arxiv.org/pdf/2004.10934.pdf)
