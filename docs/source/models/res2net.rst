Res2Net
=======

.. currentmodule:: holocron.models

The Res2Net model is based on the `"Res2Net: A New Multi-scale Backbone Architecture" <https://arxiv.org/pdf/1904.01169.pdf>`_ paper.

Architecture overview
---------------------

This paper replaces the bottleneck block of ResNet architectures by a multi-scale version.

.. image:: https://github.com/frgfm/Holocron/releases/download/v0.2.1/res2net.png
        :align: center

The key takeaways from the paper are the following:

* switch to efficient multi-scale convolutions using a cascade of conv 3x3
* adapt the block for cardinality & SE blocks


Model builders
--------------

The following model builders can be used to instantiate a Res2Net model, with or
without pre-trained weights. All the model builders internally rely on the
``holocron.models.classification.resnet.ResNet`` base class. Please refer to the `source
code
<https://github.com/frgfm/Holocron/blob/main/holocron/models/classification/res2net.py>`_ for
more details about this class.

.. autosummary::
    :toctree: generated/
    :template: function.rst

    res2net50_26w_4s
