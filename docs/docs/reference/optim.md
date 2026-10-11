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

Fusion requires contiguous FP32 parameters, gradients, and optimizer state on MPS, with zero weight decay and
`amsgrad=False`. Each parameter, gradient, and its moment buffers must use distinct storage. Shared-storage
configurations also fall back. Other configurations use ordinary AdamP, including CPU, CUDA, and PyTorch versions
without the public `torch.mps.compile_shader` API (available since PyTorch 2.7). Compilation failure also falls back before
any tensor update; execution failures propagate without retrying a partially completed update.
An execution error may leave parameters and moment buffers partly updated. Restore both the model and optimizer
from a checkpoint before continuing.

The option defaults to `False` and remains a constructor preference when loading optimizer state. AMP can still
use fusion when its parameters and unscaled gradients are FP32. Full-training accuracy equivalence is not yet
established, so keep the ordinary implementation as the default.

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
