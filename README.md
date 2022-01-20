# Holocron

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE) [![Codacy Badge](https://app.codacy.com/project/badge/Grade/1d5c892028434715834359dce09d2210)](https://www.codacy.com/gh/frgfm/Holocron/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=frgfm/Holocron&amp;utm_campaign=Badge_Grade) ![Build Status](https://github.com/frgfm/Holocron/workflows/tests/badge.svg) [![codecov](https://codecov.io/gh/frgfm/Holocron/branch/master/graph/badge.svg)](https://codecov.io/gh/frgfm/Holocron) [![Docs](https://img.shields.io/badge/docs-available-blue.svg)](https://frgfm.github.io/Holocron)   [![Pypi](https://img.shields.io/badge/pypi-v0.1.3-blue.svg)](https://pypi.org/project/pylocron/) 

Implementations of recent Deep Learning tricks in Computer Vision, easily paired up with your favorite framework and model zoo.

> **Holocrons** were information-storage [datacron](https://starwars.fandom.com/wiki/Datacron) devices used by both the [Jedi Order](https://starwars.fandom.com/wiki/Jedi_Order) and the [Sith](https://starwars.fandom.com/wiki/Sith) that contained ancient lessons or valuable information in [holographic](https://starwars.fandom.com/wiki/Hologram) form.

*Source: Wookieepedia*

*Note: support of activation mapper and model summary has been dropped and outsourced to independent packages ([torch-cam](https://github.com/frgfm/torch-cam) & [torch-scan](https://github.com/frgfm/torch-scan)) to clarify project scope.*

## Quick Tour

### PyTorch layers for every need
- Activation: [HardMish](https://github.com/digantamisra98/H-Mish), [NLReLU](https://arxiv.org/abs/1908.03682), [FReLU](https://arxiv.org/abs/2007.11824)
- Loss: [Focal Loss](https://arxiv.org/abs/1708.02002), MultiLabelCrossEntropy, [MixupLoss](https://arxiv.org/pdf/1710.09412.pdf), [ClassBalancedWrapper](https://arxiv.org/abs/1901.05555), [ComplementCrossEntropy](https://arxiv.org/abs/2009.02189), [MutualChannelLoss](https://arxiv.org/abs/2002.04264), [DiceLoss](https://arxiv.org/abs/1606.04797)
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
- Optimizer: [LARS](https://arxiv.org/abs/1708.03888), [Lamb](https://arxiv.org/abs/1904.00962), [TAdam](https://arxiv.org/pdf/2003.00179.pdf), [AdamP](https://arxiv.org/pdf/2006.08217), [AdaBelief](https://arxiv.org/abs/2010.07468), and customized versions (RaLars)
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

### Latency benchmark

You crave for SOTA performances, but you don't know whether it fits your needs in terms of latency?

In the table below, you will find a latency benchmark for all supported models:

| Arch                                                         | GPU mean (std)    | CPU mean (std)     |
| ------------------------------------------------------------ | ----------------- | ------------------ |
| [repvgg_a0](https://frgfm.github.io/Holocron/latest/models.html#holocron.models.repvgg_a0)* | 3.14ms (0.87ms)   | 23.28ms (1.21ms)   |
| [repvgg_a1](https://frgfm.github.io/Holocron/latest/models.html#holocron.models.repvgg_a1)* | 4.13ms (1.00ms)   | 29.61ms (0.46ms)   |
| [repvgg_a2](https://frgfm.github.io/Holocron/latest/models.html#holocron.models.repvgg_a2)* | 7.35ms (1.11ms)   | 46.87ms (1.27ms)   |
| [repvgg_b0](https://frgfm.github.io/Holocron/latest/models.html#holocron.models.repvgg_b0)* | 4.23ms (1.04ms)   | 33.16ms (0.58ms)   |
| [repvgg_b1](https://frgfm.github.io/Holocron/latest/models.html#holocron.models.repvgg_b1)* | 12.48ms (0.96ms)  | 100.66ms (1.46ms)  |
| [repvgg_b2](https://frgfm.github.io/Holocron/latest/models.html#holocron.models.repvgg_b2)* | 20.12ms (0.31ms)  | 155.90ms (1.59ms)  |
| [repvgg_b3](https://frgfm.github.io/Holocron/latest/models.html#holocron.models.repvgg_b3)* | 24.94ms (1.70ms)  | 224.68ms (14.27ms) |
| [rexnet1_0x](https://frgfm.github.io/Holocron/latest/models.html#holocron.models.rexnet1_0x) | 6.01ms (0.26ms)   | 13.66ms (0.21ms)   |
| [rexnet1_3x](https://frgfm.github.io/Holocron/latest/models.html#holocron.models.rexnet1_3x) | 6.43ms (0.10ms)   | 19.13ms (2.05ms)   |
| [rexnet1_5x](https://frgfm.github.io/Holocron/latest/models.html#holocron.models.rexnet1_5x) | 6.46ms (0.28ms)   | 21.06ms (0.24ms)   |
| [rexnet2_0x](https://frgfm.github.io/Holocron/latest/models.html#holocron.models.rexnet2_0x) | 6.75ms (0.21ms)   | 31.77ms (3.28ms)   |
| [rexnet2_2x](https://frgfm.github.io/Holocron/latest/models.html#holocron.models.rexnet2_2x) | 6.92ms (0.51ms)   | 33.61ms (0.60ms)   |
| [sknet50](https://frgfm.github.io/Holocron/latest/models.html#holocron.models.sknet50) | 11.40ms (0.38ms)  | 54.03ms (3.35ms)   |
| [sknet101](https://frgfm.github.io/Holocron/latest/models.html#holocron.models.sknet101) | 23.55 ms (1.11ms) | 94.89ms (5.61ms)   |
| [sknet152](https://frgfm.github.io/Holocron/latest/models.html#holocron.models.sknet152) | 69.81ms (0.60ms)  | 253.07ms (3.33ms)  |
| [tridentnet50](https://frgfm.github.io/Holocron/latest/models.html#holocron.models.tridentnet50) | 16.62ms (1.21ms)  | 142.85ms (5.33ms)  |
| [res2net50_26w_4s](https://frgfm.github.io/Holocron/latest/models.html#holocron.models.res2net50_26w_4s) | 9.25ms (0.22ms)   | 41.84ms (0.80ms)   |
| [resnet50d](https://frgfm.github.io/Holocron/latest/models.html#holocron.models.resnet50d) | 36.97ms (3.58ms)  | 36.97ms (3.58ms)   |
| [pyconv_resnet50](https://frgfm.github.io/Holocron/latest/models.html#holocron.models.pyconv_resnet50) | 20.03ms (0.28ms)  | 178.85ms (2.35ms)  |
| [pyconvhg_resnet50](https://frgfm.github.io/Holocron/latest/models.html#holocron.models.pyconvhg_resnet50) | 38.41ms (0.33ms)  | 301.03ms (12.39ms) |
| [darknet24](https://frgfm.github.io/Holocron/latest/models.html#holocron.models.darknet24) | 3.94ms (1.08ms)   | 29.39ms (0.78ms)   |
| [darknet19](https://frgfm.github.io/Holocron/latest/models.html#holocron.models.darknet19) | 3.17ms (0.59ms)   | 26.36ms (2.80ms)   |
| [darknet53](https://frgfm.github.io/Holocron/latest/models.html#holocron.models.darknet53) | 7.12ms (1.35ms)   | 53.20ms (1.17ms)   |
| [cspdarknet53](https://frgfm.github.io/Holocron/latest/models.html#holocron.models.cspdarknet53) | 6.41ms (0.21ms)   | 48.05ms (3.68ms)   |
| [cspdarknet53_mish](https://frgfm.github.io/Holocron/latest/models.html#holocron.models.cspdarknet53_mish) | 6.88ms (0.51ms)   | 67.78ms (2.90ms)   |

**The reported latency for RepVGG models is the one of the reparametrized version*

This benchmark was performed over 100 iterations on (224, 224) inputs, on a laptop to better reflect performances that can be expected by common users. The hardware setup includes an [Intel(R) Core(TM) i7-10750H](https://ark.intel.com/content/www/us/en/ark/products/201837/intel-core-i710750h-processor-12m-cache-up-to-5-00-ghz.html) for the CPU, and a [NVIDIA GeForce RTX 2070 with Max-Q Design](https://www.nvidia.com/fr-fr/geforce/graphics-cards/rtx-2070/) for the GPU.

You can run this latency benchmark for any model on your hardware as follows:

```bash
python scripts/eval_latency.py rexnet1_0x
```

*All script arguments can be checked using `python scripts/eval_latency.py --help`*



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
