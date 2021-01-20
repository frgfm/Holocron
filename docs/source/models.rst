holocron.models
###############

The models subpackage contains definitions of models for addressing
different tasks, including: image classification, pixelwise semantic
segmentation, object detection, instance segmentation, person
keypoint detection and video classification.

The following models are available:

.. contents:: Models
    :local:

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
.. autofunction:: repvgg_a0

.. autofunction:: repvgg_a1

.. autofunction:: repvgg_a2

.. autofunction:: repvgg_b0

.. autofunction:: repvgg_b1

.. autofunction:: repvgg_b2

.. autofunction:: repvgg_b3



Object Detection
================

Object detection models expect a 4D image tensor as an input (N x C x H x W) and returns a list of dictionaries.
Each dictionary has 3 keys: box coordinates, classification probability, classification label.

.. code:: python

    import holocron.models as models
    yolov2 = models.yolov2(num_classes=10)


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


U-Net
-----

.. autofunction:: unet

.. autofunction:: unetp

.. autofunction:: unetpp

.. autofunction:: unet3p
