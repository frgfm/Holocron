DarkNetV3
=========

.. currentmodule:: holocron.models

The DarkNetV3 model is based on the `"YOLOv3: An Incremental Improvement" <https://pjreddie.com/media/files/papers/YOLOv3.pdf>`_ paper.

Architecture overview
---------------------

This paper makes a more powerful version than its predecedors by increasing depth and using ResNet tricks.

The key takeaways from the paper are the following:

* adds residual connection compared to DarkNetV2


Model builders
--------------

The following model builders can be used to instantiate a DarknetV3 model, with or
without pre-trained weights. All the model builders internally rely on the
``holocron.models.classification.darknetv3.DarknetV3`` base class. Please refer to the `source
code
<https://github.com/frgfm/Holocron/blob/main/holocron/models/classification/darknetv3.py>`_ for
more details about this class.

.. autosummary::
    :toctree: generated/
    :template: function.rst

    darknet53
