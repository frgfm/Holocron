holocron.nn
============

An addition to the :mod:`torch.nn` module of Pytorch to extend the range of neural networks building blocks.


.. currentmodule:: holocron.nn

Non-linear activations
----------------------

.. autoclass:: HardMish

.. autoclass:: NLReLU

.. autoclass:: FReLU

Loss functions
--------------

.. autoclass:: FocalLoss

.. autoclass:: MultiLabelCrossEntropy

.. autoclass:: ComplementCrossEntropy

.. autoclass:: MutualChannelLoss

.. autoclass:: DiceLoss

.. autoclass:: PolyLoss


Loss wrappers
--------------

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

.. autoclass:: GlobalMaxPool2d

.. autoclass:: BlurPool2d

.. autoclass:: SPP

.. autoclass:: ZPool


Attention
---------

.. autoclass:: SAM

.. autoclass:: LambdaLayer

.. autoclass:: TripletAttention


Transformers
------------

.. autoclass:: TransformerEncoderBlock
