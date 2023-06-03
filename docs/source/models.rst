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


.. toctree::
  :caption: Supported architectures
  :maxdepth: 1

  models/resnet
  models/resnext
  models/res2net
  models/tridentnet
  models/convnext
  models/pyconv_resnet
  models/rexnet
  models/sknet
  models/darknet
  models/darknetv2
  models/darknetv3
  models/darknetv4
  models/repvgg
  models/mobileone
  models/cct


Here is the list of available checkpoints:

.. include:: generated/classification_table.rst



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
