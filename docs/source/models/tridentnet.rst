TridentNet
==========

.. currentmodule:: holocron.models

The ResNeXt model is based on the `"Scale-Aware Trident Networks for Object Detection" <https://arxiv.org/pdf/1901.01892.pdf>`_ paper.

Architecture overview
---------------------

This paper replaces the bottleneck block of ResNet architectures by a multi-scale version.

.. image:: https://github.com/frgfm/Holocron/releases/download/v0.2.1/tridentnet.png
        :align: center

The key takeaways from the paper are the following:

* switch bottleneck to a 3 branch system
* all parallel branches share the same parameters but using different dilation values


Model builders
--------------

The following model builders can be used to instantiate a TridentNet model, with or
without pre-trained weights. All the model builders internally rely on the
``holocron.models.classification.resnet.ResNet`` base class. Please refer to the `source
code
<https://github.com/frgfm/Holocron/blob/main/holocron/models/classification/tridentnet.py>`_ for
more details about this class.

.. autosummary::
    :toctree: generated/
    :template: function.rst

    tridentnet50
