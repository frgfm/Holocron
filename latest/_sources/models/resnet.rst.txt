ResNet
======

.. currentmodule:: holocron.models

The ResNet model is based on the `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_ paper.

Architecture overview
---------------------

This paper introduces a few tricks to maximize the depth of convolutional architectures that can be trained.

The key takeaways from the paper are the following:

* add a shortcut connection in bottleneck blocks to ease the gradient flow
* extensive use of batch normalization layers


Model builders
--------------

The following model builders can be used to instantiate a ResNeXt model, with or
without pre-trained weights. All the model builders internally rely on the
``holocron.models.classification.resnet.ResNet`` base class. Please refer to the `source
code
<https://github.com/frgfm/Holocron/blob/main/holocron/models/classification/resnet.py>`_ for
more details about this class.

.. autosummary::
    :toctree: generated/
    :template: function.rst

    resnet18
    resnet34
    resnet50
    resnet50d
    resnet101
    resnet152
