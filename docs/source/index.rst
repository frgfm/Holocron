*********************************************
Holocron: a Deep Learning toolbox for PyTorch
*********************************************

.. image:: https://github.com/frgfm/Holocron/releases/download/v0.1.3/holocron_logo_text.png
        :align: center

Holocron is meant to bridge the gap between PyTorch and latest research papers. It brings training components that are not available yet in PyTorch with a similar interface.

This project is meant for:

* |:zap:| **speed**: architectures in this repo are picked for both pure performances and minimal latency
* |:woman_scientist:| **research**: train your models easily to SOTA standards


.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   :hidden:

   installing
   notebooks



Model zoo
^^^^^^^^^


Image classification
""""""""""""""""""""
   * TridentNet from `"Scale-Aware Trident Networks for Object Detection" <https://arxiv.org/pdf/1901.01892.pdf>`_
   * SKNet from `"Selective Kernel Networks" <https://arxiv.org/pdf/1903.06586.pdf>`_
   * PyConvResNet from `"Pyramidal Convolution: Rethinking Convolutional Neural Networks for Visual Recognition" <https://arxiv.org/pdf/2006.11538.pdf>`_
   * ReXNet from `"ReXNet: Diminishing Representational Bottleneck on Convolutional Neural Network" <https://arxiv.org/pdf/2007.00992.pdf>`_
   * RepVGG from `"RepVGG: Making VGG-style ConvNets Great Again" <https://arxiv.org/pdf/2101.03697.pdf>`_

Semantic segmentation
"""""""""""""""""""""
   * U-Net from `"U-Net: Convolutional Networks for Biomedical Image Segmentation" <https://arxiv.org/pdf/1505.04597.pdf>`_
   * U-Net++ from `"UNet++: Redesigning Skip Connections to Exploit Multiscale Features in Image Segmentation" <https://arxiv.org/pdf/1912.05074.pdf>`_
   * UNet3+ from `"UNet 3+: A Full-Scale Connected UNet For Medical Image Segmentation" <https://arxiv.org/pdf/2004.08790.pdf>`_

Object detection
""""""""""""""""
   * YOLO from `"ou Only Look Once: Unified, Real-Time Object Detection" <https://pjreddie.com/media/files/papers/yolo_1.pdf>`_
   * YOLOv2 from `"YOLO9000: Better, Faster, Stronger" <https://pjreddie.com/media/files/papers/YOLO9000.pdf>`_
   * YOLOv4 from `"YOLOv4: Optimal Speed and Accuracy of Object Detection" <https://arxiv.org/pdf/2004.10934.pdf>`_


.. toctree::
   :maxdepth: 2
   :caption: Package Reference
   :hidden:

   models
   nn
   nn.functional
   ops
   optim
   trainer
   utils
   utils.data


.. toctree::
   :maxdepth: 2
   :caption: Notes
   :hidden:

   changelog
