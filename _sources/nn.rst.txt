holocron.nn
============

An addition to the :mod:`torch.nn` module of Pytorch to extend the range of neural networks building blocks.


.. currentmodule:: holocron.nn

Non-linear activations
----------------------

.. autoclass:: Mish

.. autoclass:: NLReLU

Loss functions
--------------

.. autoclass:: FocalLoss

.. autoclass:: MultiLabelCrossEntropy

.. autoclass:: LabelSmoothingCrossEntropy

Loss wrappers
--------------

.. autoclass:: MixupLoss

Convolution layers
------------------

.. autoclass:: NormConv2d


Downsampling
------------

.. autoclass:: ConcatDownsample2d
