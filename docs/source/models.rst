holocron.models
###############

The models subpackage contains definitions of models for addressing
different tasks, including: image classification, pixelwise semantic
segmentation, object detection, instance segmentation, person
keypoint detection and video classification.

.. currentmodule:: holocron.models


Classification
==============

Classification models expect a 4D image tensor as an input (N x C x H x W) and returns a 2D output (N x K).
The output represents the classification scores for each output classes.

.. code:: python

    import holocron.models as models
    darknet19 = models.darknet19(num_classes=10)


ResNet
-------

.. autofunction:: resnet18

.. autofunction:: resnet34

.. autofunction:: resnet50

.. autofunction:: resnet101

.. autofunction:: resnet152

.. autofunction:: resnext50_32x4d

.. autofunction:: resnext101_32x8d

.. autofunction:: resnet50d


Res2Net
-------

.. autofunction:: res2net50_26w_4s


TridentNet
----------

.. autofunction:: tridentnet50


PyConvResNet
------------

.. autofunction:: pyconv_resnet50

.. autofunction:: pyconvhg_resnet50


ReXNet
-------

.. autofunction:: rexnet1_0x

.. autofunction:: rexnet1_3x

.. autofunction:: rexnet1_5x

.. autofunction:: rexnet2_0x

.. autofunction:: rexnet2_2x


SKNet
-----

.. autofunction:: sknet50

.. autofunction:: sknet101

.. autofunction:: sknet152


Darknet
-------

.. autofunction:: darknet24

.. autofunction:: darknet19

.. autofunction:: darknet53

.. autofunction:: cspdarknet53

.. autofunction:: cspdarknet53_mish


RepVGG
------

The RepVGG architecture key aspect to have different block architectures between training and inference modes. The goal is to combine the original VGG speed and the block design of ResNet.

.. image:: https://github.com/frgfm/Holocron/releases/download/v0.2.1/repvgg.png
        :align: center

In order to do so, the block is designed in a similar fashion as a ResNet bottleneck but in a way that all branches can be fused into a single one. The more complex training architecture improves gradient flow and overall optimization, while its inference counterpart is optimized for minimum latency and memory usage. 

.. autofunction:: repvgg_a0

.. autofunction:: repvgg_a1

.. autofunction:: repvgg_a2

.. autofunction:: repvgg_b0

.. autofunction:: repvgg_b1

.. autofunction:: repvgg_b2

.. autofunction:: repvgg_b3


ConvNeXt
--------

The ConvNeXt architecture compiles tricks from transformer-based vision models to improve a pure convolutional model.

.. image:: https://github.com/frgfm/Holocron/releases/download/v0.2.1/convnext.png
        :align: center

The key takeaways from the paper are the following:
* update the stem convolution to act like a patchify layer of transformers
* increase block kernel size to 7
* switch to depth-wise convolutions
* reduce the amount of activations and normalization layers

.. autofunction:: convnext_micro

.. autofunction:: convnext_tiny

.. autofunction:: convnext_small

.. autofunction:: convnext_base

.. autofunction:: convnext_large

.. autofunction:: convnext_xl


MobileOne
---------

The MobileOne architecture key takeaway is to optimize the model for inference speed at inference time on mobile device. It reuses the reparametrization concept of RepVGG while adding overparametrization in the block branches.

.. image:: https://github.com/frgfm/Holocron/releases/download/v0.2.1/mobileone.png
        :align: center

Each block is composed of two consecutive reparametrizeable blocks (in a similar fashion than RepVGG):
* a depth-wise convolutional block
* a point-wise convolutional block

.. autofunction:: mobileone_s0

.. autofunction:: mobileone_s1

.. autofunction:: mobileone_s2

.. autofunction:: mobileone_s3


Object Detection
================

Object detection models expect a 4D image tensor as an input (N x C x H x W) and returns a list of dictionaries.
Each dictionary has 3 keys: box coordinates, classification probability, classification label.

.. code:: python

    import holocron.models as models
    yolov2 = models.yolov2(num_classes=10)


.. currentmodule:: holocron.models.detection

YOLO
----

.. autofunction:: yolov1

.. autofunction:: yolov2

.. autofunction:: yolov4


Semantic Segmentation
=====================

Semantic segmentation models expect a 4D image tensor as an input (N x C x H x W) and returns a classification score
tensor of size (N x K x Ho x Wo).

.. code:: python

    import holocron.models as models
    unet = models.unet(num_classes=10)


.. currentmodule:: holocron.models.segmentation


U-Net
-----

.. autofunction:: unet

.. autofunction:: unetp

.. autofunction:: unetpp

.. autofunction:: unet3p

.. autofunction:: unet2

.. autofunction:: unet_tvvgg11

.. autofunction:: unet_tvresnet34

.. autofunction:: unet_rexnet13
