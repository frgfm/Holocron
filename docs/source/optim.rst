holocron.optim
===============

.. automodule:: holocron.optim

.. currentmodule:: holocron.optim

To use :mod:`holocron.optim` you have to construct an optimizer object, that will hold
the current state and will update the parameters based on the computed gradients.

Optimizers
----------

.. autoclass:: Lamb
    :members:
.. autoclass:: Lars
    :members:
.. autoclass:: Lookahead
    :members:
.. autoclass:: RAdam
    :members:
.. autoclass:: RaLars
    :members:


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
    :members:
