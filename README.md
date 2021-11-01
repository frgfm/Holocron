# Holocron

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE) [![Codacy Badge](https://app.codacy.com/project/badge/Grade/1d5c892028434715834359dce09d2210)](https://www.codacy.com/gh/frgfm/Holocron/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=frgfm/Holocron&amp;utm_campaign=Badge_Grade) ![Build Status](https://github.com/frgfm/Holocron/workflows/tests/badge.svg) [![codecov](https://codecov.io/gh/frgfm/Holocron/branch/master/graph/badge.svg)](https://codecov.io/gh/frgfm/Holocron) [![Docs](https://img.shields.io/badge/docs-available-blue.svg)](https://frgfm.github.io/Holocron)   [![Pypi](https://img.shields.io/badge/pypi-v0.1.3-blue.svg)](https://pypi.org/project/pylocron/) 

Implementations of recent Deep Learning tricks in Computer Vision, easily paired up with your favorite framework and model zoo.

> **Holocrons** were information-storage [datacron](https://starwars.fandom.com/wiki/Datacron) devices used by both the [Jedi Order](https://starwars.fandom.com/wiki/Jedi_Order) and the [Sith](https://starwars.fandom.com/wiki/Sith) that contained ancient lessons or valuable information in [holographic](https://starwars.fandom.com/wiki/Hologram) form.

*Source: Wookieepedia*

*Note: support of activation mapper and model summary has been dropped and outsourced to independent packages ([torch-cam](https://github.com/frgfm/torch-cam) & [torch-scan](https://github.com/frgfm/torch-scan)) to clarify project scope.*

## Quick Tour

### PyTorch layers for every need
- Activation: [SiLU/Swish](https://arxiv.org/abs/1606.08415), [Mish](https://arxiv.org/abs/1908.08681), [HardMish](https://github.com/digantamisra98/H-Mish), [NLReLU](https://arxiv.org/abs/1908.03682), [FReLU](https://arxiv.org/abs/2007.11824)
- Loss: [Focal Loss](https://arxiv.org/abs/1708.02002), MultiLabelCrossEntropy, [LabelSmoothingCrossEntropy](https://arxiv.org/pdf/1706.03762.pdf), [MixupLoss](https://arxiv.org/pdf/1710.09412.pdf), [ClassBalancedWrapper](https://arxiv.org/abs/1901.05555), [ComplementCrossEntropy](https://arxiv.org/abs/2009.02189), [MutualChannelLoss](https://arxiv.org/abs/2002.04264)
- Convolutions: [NormConv2d](https://arxiv.org/pdf/2005.05274v2.pdf), [Add2d](https://arxiv.org/pdf/1912.13200.pdf), [SlimConv2d](https://arxiv.org/pdf/2003.07469.pdf), [PyConv2d](https://arxiv.org/abs/2006.11538), [Involution](https://arxiv.org/abs/2103.06255)
- Regularization: [DropBlock](https://arxiv.org/abs/1810.12890)
- Pooling: [BlurPool2d](https://arxiv.org/abs/1904.11486), [SPP](https://arxiv.org/abs/1406.4729), [ZPool](https://arxiv.org/abs/2010.03045)
- Attention: [SAM](https://arxiv.org/abs/1807.06521), [LambdaLayer](https://openreview.net/forum?id=xTJEN-ggl1b), [TripletAttention](https://arxiv.org/abs/2010.03045)

### Models for vision tasks
- Classification: [Res2Net](https://arxiv.org/abs/1904.01169) (based on the great [implementation](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/res2net.py) from Ross Wightman), [Darknet-24](https://pjreddie.com/media/files/papers/yolo_1.pdf), [Darknet-19](https://pjreddie.com/media/files/papers/YOLO9000.pdf), [Darknet-53](https://pjreddie.com/media/files/papers/YOLOv3.pdf), [CSPDarknet-53](<https://arxiv.org/abs/1911.11929>), [ResNet](https://arxiv.org/abs/1512.03385), [ResNeXt](https://arxiv.org/abs/1611.05431), [TridentNet](https://arxiv.org/abs/1901.01892), [PyConvResNet](https://arxiv.org/abs/2006.11538), [ReXNet](https://arxiv.org/abs/2007.00992), [SKNet](https://arxiv.org/abs/1903.06586), [RepVGG](https://arxiv.org/abs/2101.03697).
- Detection: [YOLOv1](https://pjreddie.com/media/files/papers/yolo_1.pdf), [YOLOv2](https://pjreddie.com/media/files/papers/YOLO9000.pdf), [YOLOv4](https://arxiv.org/abs/2004.10934)
- Segmentation: [U-Net](https://arxiv.org/abs/1505.04597), [UNet++](https://arxiv.org/abs/1807.10165), [UNet3+](https://arxiv.org/abs/2004.08790)

### Vision-related operations
- boxes: [Distance-IoU & Complete-IoU losses](https://arxiv.org/abs/1911.08287)

### Trying something else than Adam
- Optimizer: [LARS](https://arxiv.org/abs/1708.03888), [Lamb](https://arxiv.org/abs/1904.00962), [RAdam](https://arxiv.org/abs/1908.03265), [TAdam](https://arxiv.org/pdf/2003.00179.pdf), [AdamP](https://arxiv.org/pdf/2006.08217), [AdaBelief](https://arxiv.org/abs/2010.07468), and customized versions (RaLars)
- Optimizer wrapper: [Lookahead](https://arxiv.org/abs/1907.08610), Scout (experimental)



## Setup

Python 3.6 (or higher) and [pip](https://pip.pypa.io/en/stable/)/[conda](https://docs.conda.io/en/latest/miniconda.html) are required to install Holocron.

### Stable release

You can install the last stable release of the package using [pypi](https://pypi.org/project/pylocron/) as follows:

```shell
pip install pylocron
```

or using [conda](https://anaconda.org/frgfm/pylocron):

```shell
conda install -c frgfm pylocron
```

### Developer installation

Alternatively, if you wish to use the latest features of the project that haven't made their way to a release yet, you can install the package from source:

```shell
git clone https://github.com/frgfm/Holocron.git
pip install -e Holocron/.
```


## What else

### Documentation

The full package documentation is available [here](https://frgfm.github.io/holocron/) for detailed specifications.


### Reference scripts

Reference scripts are provided to train your models using holocron on famous public datasets. Those scripts currently support the following vision tasks:
- [Image classification](references/classification)
- [Object detection](references/detection)
- [Semantic segmentation](references/segmentation)



## Citation

If you wish to cite this project, feel free to use this [BibTeX](http://www.bibtex.org/) reference:

```bibtex
@misc{holocron2019,
    title={Holocron},
    author={François-Guillaume Fernandez},
    year={2019},
    month={August},
    publisher = {GitHub},
    howpublished = {\url{https://github.com/frgfm/Holocron}}
}
```


## Contributing

Any sort of contribution is greatly appreciated!

You can find a short guide in [`CONTRIBUTING`](CONTRIBUTING) to help grow this project!



## License

Distributed under the Apache 2.0 License. See [`LICENSE`](LICENSE) for more information.
