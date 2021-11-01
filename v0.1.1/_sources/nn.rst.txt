holocron.nn
============

An addition to the :mod:`torch.nn` module of Pytorch to extend the range of neural networks building blocks.


.. currentmodule:: holocron.nn

Non-linear activations
----------------------

.. autoclass:: SiLU

.. autoclass:: Mish

.. autoclass:: HardMish

.. autoclass:: NLReLU

.. autoclass:: FReLU

Loss functions
--------------

.. autoclass:: FocalLoss

.. autoclass:: MultiLabelCrossEntropy

.. autoclass:: LabelSmoothingCrossEntropy

.. autoclass:: ComplementCrossEntropy

.. autoclass:: MutualChannelLoss


Loss wrappers
--------------

.. autoclass:: MixupLoss

.. autoclass:: ClassBalancedWrapper

Convolution layers
------------------

.. autoclass:: NormConv2d

.. autoclass:: Add2d

.. autoclass:: SlimConv2d

.. autoclass:: PyConv2d

.. autoclass:: Involution2d

Regularization layers
---------------------

.. autoclass:: DropBlock2d


Downsampling
------------

.. autoclass:: ConcatDownsample2d

.. autoclass:: GlobalAvgPool2d

.. autoclass:: BlurPool2d

.. autoclass:: SPP

.. autoclass:: ZPool


Attention
---------

.. autoclass:: SAM

.. autoclass:: LambdaLayer

.. autoclass:: TripletAttention
