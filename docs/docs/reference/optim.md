# holocron.optim

To use `holocron.optim` you have to construct an optimizer object, that will hold
the current state and will update the parameters based on the computed gradients.

## Optimizers

Implementations of recent parameter optimizer for Pytorch modules.

::: holocron.optim.LARS
    options:
        heading_level: 3
        members: no

::: holocron.optim.LAMB
    options:
        heading_level: 3
        members: no

::: holocron.optim.RaLars
    options:
        heading_level: 3
        members: no

::: holocron.optim.TAdam
    options:
        heading_level: 3
        members: no

::: holocron.optim.AdaBelief
    options:
        heading_level: 3
        members: no

::: holocron.optim.AdamP
    options:
        heading_level: 3
        members: no

AdamP offers optional Metal kernel fusion on Apple Silicon:

```python
optimizer = holocron.optim.AdamP(model.parameters(), mps_fused=True)
```

A parameter group uses fusion only with contiguous FP32 MPS parameters, gradients, and state, zero weight decay,
and `amsgrad=False`. Each parameter, its gradient, and its moment buffers must use distinct storage.
Unsupported groups, missing `torch.mps.compile_shader`, and compilation failures use ordinary AdamP.
Execution errors propagate without retrying and can leave parameters and moments partly updated. Restore both
the model and optimizer from a checkpoint before continuing.

`mps_fused` defaults to `False` and remains a constructor preference when loading optimizer state. Mixed precision
can still use fusion when parameters and unscaled gradients are FP32. Full-training accuracy equivalence remains
unverified.

::: holocron.optim.Adan
    options:
        heading_level: 3
        members: no

::: holocron.optim.AdEMAMix
    options:
        heading_level: 3
        members: no


## Optimizer wrappers

`holocron.optim` also implements optimizer wrappers.

A base optimizer should always be passed to the wrapper; e.g., you
should write your code this way:

```python
optimizer = ...
optimizer = wrapper(optimizer)
```

::: holocron.optim.Lookahead
    options:
        heading_level: 3
        members: no

::: holocron.optim.Scout
    options:
        heading_level: 3
        members: no
