holocron.optim
===============

.. automodule:: holocron.optim

.. currentmodule:: holocron.optim

To use :mod:`holocron.optim` you have to construct an optimizer object, that will hold
the current state and will update the parameters based on the computed gradients.

Optimizers
----------

Implementations of recent parameter optimizer for Pytorch modules.

.. autoclass:: Lamb

.. autoclass:: Lars

.. autoclass:: RAdam

.. autoclass:: RaLars


Optimizer wrappers
------------------

:mod:`holocron.optim` also implements optimizer wrappers.

A base optimizer should always be passed to the wrapper; e.g., you
should write your code this way:

    >>> optimizer = ...
    >>> optimizer = wrapper(optimizer)

.. autoclass:: holocron.optim.wrapper.Lookahead

.. autoclass:: holocron.optim.wrapper.Scout


Learning rate schedulers
---------------------------

:mod:`holocron.optim.lr_scheduler` provides several methods to adjust the learning
rate based on the number of epochs. :class:`holocron.optim.lr_scheduler.OneCycleScheduler`
allows dynamic learning rate reducing based on some validation measurements.

Learning rate scheduling should be applied after optimizer's update; e.g., you
should write your code this way:

    >>> scheduler = ...
    >>> for epoch in range(100):
    >>>     train(...)
    >>>     validate(...)
    >>>     scheduler.step()

.. autoclass:: holocron.optim.lr_scheduler.OneCycleScheduler

