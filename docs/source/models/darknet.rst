DarkNet
=======

.. currentmodule:: holocron.models

The DarkNet model is based on the `"You Only Look Once: Unified, Real-Time Object Detection" <https://pjreddie.com/media/files/papers/yolo_1.pdf>`_ paper.

Architecture overview
---------------------

This paper introduces a highway network with powerful feature representation abilities.

The key takeaways from the paper are the following:

* improves the Inception architecture by using conv1x1
* replaces ReLU by LeakyReLU


Model builders
--------------

The following model builders can be used to instantiate a DarknetV1 model, with or
without pre-trained weights. All the model builders internally rely on the
``holocron.models.classification.darknet.DarknetV1`` base class. Please refer to the `source
code
<https://github.com/frgfm/Holocron/blob/main/holocron/models/classification/darknet.py>`_ for
more details about this class.

.. autosummary::
    :toctree: generated/
    :template: function.rst

    darknet24
