PyConvResNet
============

.. currentmodule:: holocron.models

The PyConvResNet model is based on the `"Pyramidal Convolution: Rethinking Convolutional Neural Networks for Visual Recognition" <https://arxiv.org/pdf/2006.11538.pdf>`_ paper.

Architecture overview
---------------------

This paper explores an alternative approach for convolutional block in a pyramidal fashion.

.. image:: https://github.com/frgfm/Holocron/releases/download/v0.2.1/pyconv_resnet.png
        :align: center

The key takeaways from the paper are the following:

* replaces standard convolutions with pyramidal convolutions
* extends kernel size while increasing group size to balance the number of operations


Model builders
--------------

The following model builders can be used to instantiate a PyConvResNet model, with or
without pre-trained weights. All the model builders internally rely on the
``holocron.models.classification.resnet.ResNet`` base class. Please refer to the `source
code
<https://github.com/frgfm/Holocron/blob/main/holocron/models/classification/pyconv_resnet.py>`_ for
more details about this class.

.. autosummary::
    :toctree: generated/
    :template: function.rst

    pyconv_resnet50
    pyconvhg_resnet50
