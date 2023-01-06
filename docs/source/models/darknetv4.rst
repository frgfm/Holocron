DarkNetV4
=========

.. currentmodule:: holocron.models

The DarkNetV4 model is based on the `"CSPNet: A New Backbone that can Enhance Learning Capability of CNN" <https://arxiv.org/pdf/1911.11929.pdf>`_ paper.

Architecture overview
---------------------

This paper makes a more powerful version than its predecedors by increasing depth and using ResNet tricks.

The key takeaways from the paper are the following:

* add cross-path connections to its predecessors
* explores newer non-linearities


Model builders
--------------

The following model builders can be used to instantiate a DarknetV3 model, with or
without pre-trained weights. All the model builders internally rely on the
``holocron.models.classification.darknetv4.DarknetV4`` base class. Please refer to the `source
code
<https://github.com/frgfm/Holocron/blob/main/holocron/models/classification/darknetv4.py>`_ for
more details about this class.

.. autosummary::
    :toctree: generated/
    :template: function.rst

    cspdarknet53
    cspdarknet53_mish
