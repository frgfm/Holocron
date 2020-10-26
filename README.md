# Holocron

[![License](https://img.shields.io/badge/License-MIT-brightgreen.svg)](LICENSE) [![Codacy Badge](https://api.codacy.com/project/badge/Grade/5713eafaf8074e27a4013dbfcfad9d69)](https://www.codacy.com/manual/fg/Holocron?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=frgfm/Holocron&amp;utm_campaign=Badge_Grade) ![Build Status](https://github.com/frgfm/Holocron/workflows/python-package/badge.svg) [![codecov](https://codecov.io/gh/frgfm/Holocron/branch/master/graph/badge.svg)](https://codecov.io/gh/frgfm/Holocron) [![Docs](https://img.shields.io/badge/docs-available-blue.svg)](https://frgfm.github.io/Holocron)

Implementations of recent Deep Learning tricks in Computer Vision, easily paired up with your favorite framework and model zoo.

> **Holocrons** were information-storage [datacron](https://starwars.fandom.com/wiki/Datacron) devices used by both the [Jedi Order](https://starwars.fandom.com/wiki/Jedi_Order) and the [Sith](https://starwars.fandom.com/wiki/Sith) that contained ancient lessons or valuable information in [holographic](https://starwars.fandom.com/wiki/Hologram) form.

*Source: Wookieepedia*



## Table of Contents

- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Technical Roadmap](#technical-roadmap)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)



*Note: support of activation mapper and model summary has been dropped and outsourced to independent packages ([torch-cam](https://github.com/frgfm/torch-cam) & [torch-scan](https://github.com/frgfm/torch-scan)) to clarify project scope.*



## Getting started

### Prerequisites

- Python 3.6 (or more recent)
- [pip](https://pip.pypa.io/en/stable/)

### Installation

You can install the package using [pypi](https://pypi.org/project/pylocronn/) as follows:

```bash
pip install pylocron
```

or using [conda](https://anaconda.org/frgfm/pylocron):

```bash
conda install -c frgfm pylocron
```



## Usage

### nn

##### Main features

- Activation: [SiLU/Swish](https://arxiv.org/abs/1606.08415), [Mish](https://arxiv.org/abs/1908.08681), [HardMish](https://github.com/digantamisra98/H-Mish), [NLReLU](https://arxiv.org/abs/1908.03682), [FReLU](https://arxiv.org/abs/2007.11824)
- Loss: [Focal Loss](https://arxiv.org/abs/1708.02002), MultiLabelCrossEntropy, [LabelSmoothingCrossEntropy](https://arxiv.org/pdf/1706.03762.pdf), [MixupLoss](https://arxiv.org/pdf/1710.09412.pdf), [ClassBalancedWrapper](https://arxiv.org/abs/1901.05555), [ComplementCrossEntropy](https://arxiv.org/abs/2009.02189), [MutualChannelLoss](https://arxiv.org/abs/2002.04264)
- Convolutions: [NormConv2d](https://arxiv.org/pdf/2005.05274v2.pdf), [Add2d](https://arxiv.org/pdf/1912.13200.pdf), [SlimConv2d](https://arxiv.org/pdf/2003.07469.pdf), [PyConv2d](https://arxiv.org/abs/2006.11538)
- Regularization: [DropBlock](https://arxiv.org/abs/1810.12890)
- Pooling: [BlurPool2d](https://arxiv.org/abs/1904.11486), [SPP](https://arxiv.org/abs/1406.4729)
- Attention: [SAM](https://arxiv.org/abs/1807.06521), [LambdaLayer](https://openreview.net/forum?id=xTJEN-ggl1b)

### models

##### Main features

- Classification: [Res2Net](https://arxiv.org/abs/1904.01169) (based on the great [implementation](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/res2net.py) from Ross Wightman), [Darknet-24](https://pjreddie.com/media/files/papers/yolo_1.pdf), [Darknet-19](https://pjreddie.com/media/files/papers/YOLO9000.pdf), [Darknet-53](https://pjreddie.com/media/files/papers/YOLOv3.pdf), [CSPDarknet-53](<https://arxiv.org/abs/1911.11929>), [ResNet](https://arxiv.org/abs/1512.03385), [ResNeXt](https://arxiv.org/abs/1611.05431), [TridentNet](https://arxiv.org/abs/1901.01892), [PyConvResNet](https://arxiv.org/abs/2006.11538), [ReXNet](https://arxiv.org/abs/2007.00992), [SKNet](https://arxiv.org/abs/1903.06586).
- Detection: [YOLOv1](https://pjreddie.com/media/files/papers/yolo_1.pdf), [YOLOv2](https://pjreddie.com/media/files/papers/YOLO9000.pdf), [YOLOv4](https://arxiv.org/abs/2004.10934)
- Segmentation: [U-Net](https://arxiv.org/abs/1505.04597), [UNet++](https://arxiv.org/abs/1807.10165), [UNet3+](https://arxiv.org/abs/2004.08790)

### ops

##### Main features

- boxes: [Distance-IoU & Complete-IoU losses](https://arxiv.org/abs/1911.08287)

### optim

##### Main features

- Optimizer: [LARS](https://arxiv.org/abs/1708.03888), [Lamb](https://arxiv.org/abs/1904.00962), [RAdam](https://arxiv.org/abs/1908.03265), [TAdam](https://arxiv.org/pdf/2003.00179.pdf), [AdaBelief](https://arxiv.org/abs/2010.07468), and customized versions (RaLars)
- Optimizer wrapper: [Lookahead](https://arxiv.org/abs/1907.08610), Scout (experimental)
- Scheduler: [OneCycleScheduler](https://arxiv.org/abs/1803.09820) *(this implementation was made before PyTorch officially had an [implementation](https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.OneCycleLR), for better support it is advised to consider the official PyTorch version)*



## Technical roadmap

The project is currently under development, here are the objectives for the next releases:

- [x] Standardize models: standardize models by task.
- [x] Reference scripts: add reference training scripts
- [ ] Speed benchmark: compare `holocron.nn` functions execution speed.



## Documentation

The full package documentation is available [here](<https://frgfm.github.io/Holocron/>) for detailed specifications. The documentation was built with [Sphinx](sphinx-doc.org) using a [theme](github.com/readthedocs/sphinx_rtd_theme) provided by [Read the Docs](readthedocs.org) 



## Contributing

Please refer to `CONTRIBUTING` if you wish to contribute to this project.



## License

Distributed under the MIT License. See `LICENSE` for more information.