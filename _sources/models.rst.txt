holocron.models
###############

The models subpackage contains definitions of models for addressing
different tasks, including: image classification, pixelwise semantic
segmentation, object detection, instance segmentation, person
keypoint detection and video classification.

The following datasets are available:

.. contents:: Models
    :local:

.. currentmodule:: holocron.models


Classification
==============

Classification models expects a 4D image tensor as an input (N x C x H x W) and returns a 2D output (N x K).
The output represents the classification scores for each output classes.

.. code:: python

    import holocron.models as models
    darknet19 = models.darknet19(num_classes=10)


Res2Net
-------

.. autoclass:: Res2Net

.. autofunction:: res2net

Res2NeXt
--------

.. autofunction:: res2next


Darknet
-------

.. autofunction:: darknet19

