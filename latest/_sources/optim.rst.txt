holocron.optim
===============

.. automodule:: holocron.optim

.. currentmodule:: holocron.optim

To use :mod:`holocron.optim` you have to construct an optimizer object, that will hold
the current state and will update the parameters based on the computed gradients.

Optimizers
----------

Implementations of recent parameter optimizer for Pytorch modules.

.. autoclass:: LARS

.. autoclass:: LAMB

.. autoclass:: RaLars

.. autoclass:: TAdam

.. autoclass:: AdaBelief

.. autoclass:: AdamP

.. autoclass:: Adan

.. autoclass:: AdEMAMix


Optimizer wrappers
------------------

:mod:`holocron.optim` also implements optimizer wrappers.

A base optimizer should always be passed to the wrapper; e.g., you
should write your code this way:

    >>> optimizer = ...
    >>> optimizer = wrapper(optimizer)

.. autoclass:: holocron.optim.wrapper.Lookahead

.. autoclass:: holocron.optim.wrapper.Scout
