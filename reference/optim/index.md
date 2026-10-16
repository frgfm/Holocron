# holocron.optim

To use `holocron.optim` you have to construct an optimizer object, that will hold the current state and will update the parameters based on the computed gradients.

## Optimizers

Implementations of recent parameter optimizer for Pytorch modules.

### LARS

```python
LARS(params: Iterable[Parameter], lr: float = 0.001, momentum: float = 0.0, dampening: float = 0.0, weight_decay: float = 0.0, nesterov: bool = False, scale_clip: tuple[float, float] | None = None)
```

Bases: `Optimizer`

Implements the LARS optimizer from ["Large batch training of convolutional networks"](https://arxiv.org/pdf/1708.03888.pdf).

The estimation of global and local learning rates is described as follows, (\\forall t \\geq 1):

[ \\alpha_t \\leftarrow \\alpha (1 - t / T)^2 \\ \\gamma_t \\leftarrow \\frac{\\lVert \\theta_t \\rVert}{\\lVert g_t \\rVert + \\lambda \\lVert \\theta_t \\rVert} ]

where (\\theta_t) is the parameter value at step (t) ((\\theta_0) being the initialization value), (g_t) is the gradient of (\\theta_t), (T) is the total number of steps, (\\alpha) is the learning rate (\\lambda \\geq 0) is the weight decay.

Then we estimate the momentum using:

[ v_t \\leftarrow m v\_{t-1} + \\alpha_t \\gamma_t (g_t + \\lambda \\theta_t) ]

where (m) is the momentum and (v_0 = 0).

And finally the update step is performed using the following rule:

[ \\theta_t \\leftarrow \\theta\_{t-1} - v_t ]

| PARAMETER      | DESCRIPTION                                                                                           |
| -------------- | ----------------------------------------------------------------------------------------------------- |
| `params`       | iterable of parameters to optimize or dicts defining parameter groups **TYPE:** `Iterable[Parameter]` |
| `lr`           | learning rate **TYPE:** `float` **DEFAULT:** `0.001`                                                  |
| `momentum`     | momentum factor **TYPE:** `float` **DEFAULT:** `0.0`                                                  |
| `weight_decay` | weight decay (L2 penalty) **TYPE:** `float` **DEFAULT:** `0.0`                                        |
| `dampening`    | dampening for momentum **TYPE:** `float` **DEFAULT:** `0.0`                                           |
| `nesterov`     | enables Nesterov momentum **TYPE:** `bool` **DEFAULT:** `False`                                       |
| `scale_clip`   | the lower and upper bounds for the weight norm in local LR of LARS **TYPE:** \`tuple[float, float]    |

Source code in `holocron/optim/lars.py`

```python
def __init__(
    self,
    params: Iterable[torch.nn.parameter.Parameter],
    lr: float = 1e-3,
    momentum: float = 0.0,
    dampening: float = 0.0,
    weight_decay: float = 0.0,
    nesterov: bool = False,
    scale_clip: tuple[float, float] | None = None,
) -> None:
    if not isinstance(lr, float) or lr < 0.0:
        raise ValueError(f"Invalid learning rate: {lr}")
    if momentum < 0.0:
        raise ValueError(f"Invalid momentum value: {momentum}")
    if weight_decay < 0.0:
        raise ValueError(f"Invalid weight_decay value: {weight_decay}")

    defaults = {
        "lr": lr,
        "momentum": momentum,
        "dampening": dampening,
        "weight_decay": weight_decay,
        "nesterov": nesterov,
    }
    if nesterov and (momentum <= 0 or dampening != 0):
        raise ValueError("Nesterov momentum requires a momentum and zero dampening")
    super().__init__(params, defaults)
    # LARS arguments
    self.scale_clip = scale_clip
    if self.scale_clip is None:
        self.scale_clip = (0.0, 10.0)
```

### LAMB

```python
LAMB(params: Iterable[Parameter], lr: float = 0.001, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-08, weight_decay: float = 0.0, scale_clip: tuple[float, float] | None = None)
```

Bases: `Optimizer`

Implements the Lamb optimizer from ["Large batch optimization for deep learning: training BERT in 76 minutes"](https://arxiv.org/pdf/1904.00962v3.pdf).

The estimation of momentums is described as follows, (\\forall t \\geq 1):

[ m_t \\leftarrow \\beta_1 m\_{t-1} + (1 - \\beta_1) g_t \\ v_t \\leftarrow \\beta_2 v\_{t-1} + (1 - \\beta_2) g_t^2 ]

where (g_t) is the gradient of (\\theta_t), (\\beta_1, \\beta_2 \\in [0, 1]^2) are the exponential average smoothing coefficients, (m_0 = 0,\\ v_0 = 0).

Then we correct their biases using:

[ \\hat{m_t} \\leftarrow \\frac{m_t}{1 - \\beta_1^t} \\ \\hat{v_t} \\leftarrow \\frac{v_t}{1 - \\beta_2^t} ]

And finally the update step is performed using the following rule:

[ r_t \\leftarrow \\frac{\\hat{m_t}}{\\sqrt{\\hat{v_t}} + \\epsilon} \\ \\theta_t \\leftarrow \\theta\_{t-1} - \\alpha \\phi(\\lVert \\theta_t \\rVert) \\frac{r_t + \\lambda \\theta_t}{\\lVert r_t + \\theta_t \\rVert} ]

where (\\theta_t) is the parameter value at step (t) ((\\theta_0) being the initialization value), (\\phi) is a clipping function, (\\alpha) is the learning rate, (\\lambda \\geq 0) is the weight decay, (\\epsilon > 0).

| PARAMETER      | DESCRIPTION                                                                                             |
| -------------- | ------------------------------------------------------------------------------------------------------- |
| `params`       | iterable of parameters to optimize or dicts defining parameter groups **TYPE:** `Iterable[Parameter]`   |
| `lr`           | learning rate **TYPE:** `float` **DEFAULT:** `0.001`                                                    |
| `betas`        | beta coefficients used for running averages **TYPE:** `tuple[float, float]` **DEFAULT:** `(0.9, 0.999)` |
| `eps`          | term added to the denominator to improve numerical stability **TYPE:** `float` **DEFAULT:** `1e-08`     |
| `weight_decay` | weight decay (L2 penalty) **TYPE:** `float` **DEFAULT:** `0.0`                                          |
| `scale_clip`   | the lower and upper bounds for the weight norm in local LR of LARS **TYPE:** \`tuple[float, float]      |

Source code in `holocron/optim/lamb.py`

```python
def __init__(
    self,
    params: Iterable[torch.nn.Parameter],
    lr: float = 1e-3,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    scale_clip: tuple[float, float] | None = None,
) -> None:
    if lr < 0.0:
        raise ValueError(f"Invalid learning rate: {lr}")
    if eps < 0.0:
        raise ValueError(f"Invalid epsilon value: {eps}")
    if not 0.0 <= betas[0] < 1.0:
        raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
    if not 0.0 <= betas[1] < 1.0:
        raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
    defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
    super().__init__(params, defaults)
    # LARS arguments
    self.scale_clip = scale_clip
    if self.scale_clip is None:
        self.scale_clip = (0.0, 10.0)
```

### RaLars

```python
RaLars(params: Iterable[Parameter], lr: float = 0.001, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-08, weight_decay: float = 0.0, force_adaptive_momentum: bool = False, scale_clip: tuple[float, float] | None = None)
```

Bases: `Optimizer`

Implements the RAdam optimizer from ["On the variance of the Adaptive Learning Rate and Beyond"](https://arxiv.org/pdf/1908.03265.pdf) with optional Layer-wise adaptive Scaling from ["Large Batch Training of Convolutional Networks"](https://arxiv.org/pdf/1708.03888.pdf)

| PARAMETER                 | DESCRIPTION                                                                                           |
| ------------------------- | ----------------------------------------------------------------------------------------------------- |
| `params`                  | iterable of parameters to optimize or dicts defining parameter groups **TYPE:** `Iterable[Parameter]` |
| `lr`                      | learning rate **TYPE:** `float` **DEFAULT:** `0.001`                                                  |
| `betas`                   | coefficients used for running averages **TYPE:** `tuple[float, float]` **DEFAULT:** `(0.9, 0.999)`    |
| `eps`                     | term added to the denominator to improve numerical stability **TYPE:** `float` **DEFAULT:** `1e-08`   |
| `weight_decay`            | weight decay (L2 penalty) **TYPE:** `float` **DEFAULT:** `0.0`                                        |
| `force_adaptive_momentum` | use adaptive momentum if variance is not tractable **TYPE:** `bool` **DEFAULT:** `False`              |
| `scale_clip`              | the maximal upper bound for the scale factor of LARS **TYPE:** \`tuple[float, float]                  |

Source code in `holocron/optim/ralars.py`

```python
def __init__(
    self,
    params: Iterable[torch.nn.Parameter],
    lr: float = 1e-3,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    force_adaptive_momentum: bool = False,
    scale_clip: tuple[float, float] | None = None,
) -> None:
    if lr < 0.0:
        raise ValueError(f"Invalid learning rate: {lr}")
    if eps < 0.0:
        raise ValueError(f"Invalid epsilon value: {eps}")
    if not 0.0 <= betas[0] < 1.0:
        raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
    if not 0.0 <= betas[1] < 1.0:
        raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
    defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
    super().__init__(params, defaults)
    # RAdam tweaks
    self.force_adaptive_momentum = force_adaptive_momentum
    # LARS arguments
    self.scale_clip = scale_clip
    if self.scale_clip is None:
        self.scale_clip = (0, 10)
```

### TAdam

```python
TAdam(params: Iterable[Parameter], lr: float = 0.001, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-08, weight_decay: float = 0.0, amsgrad: bool = False, dof: float | None = None)
```

Bases: `Optimizer`

Implements the TAdam optimizer from ["TAdam: A Robust Stochastic Gradient Optimizer"](https://arxiv.org/pdf/2003.00179.pdf).

The estimation of momentums is described as follows, (\\forall t \\geq 1):

[ w_t \\leftarrow (\\nu + d) \\Big(\\nu + \\sum\\limits\_{j} \\frac{(g_t^j - m\_{t-1}^j)^2}{v\_{t-1} + \\epsilon} \\Big)^{-1} \\ m_t \\leftarrow \\frac{W\_{t-1}}{W\_{t-1} + w_t} m\_{t-1} + \\frac{w_t}{W\_{t-1} + w_t} g_t \\ v_t \\leftarrow \\beta_2 v\_{t-1} + (1 - \\beta_2) (g_t - g\_{t-1}) ]

where (g_t) is the gradient of (\\theta_t), (\\beta_1, \\beta_2 \\in [0, 1]^2) are the exponential average smoothing coefficients, (m_0 = 0,\\ v_0 = 0,\\ W_0 = \\frac{\\beta_1}{1 - \\beta_1}); (\\nu) is the degrees of freedom and (d) if the number of dimensions of the parameter gradient.

Then we correct their biases using:

[ \\hat{m_t} \\leftarrow \\frac{m_t}{1 - \\beta_1^t} \\ \\hat{v_t} \\leftarrow \\frac{v_t}{1 - \\beta_2^t} ]

And finally the update step is performed using the following rule:

[ \\theta_t \\leftarrow \\theta\_{t-1} - \\alpha \\frac{\\hat{m_t}}{\\sqrt{\\hat{v_t}} + \\epsilon} ]

where (\\theta_t) is the parameter value at step (t) ((\\theta_0) being the initialization value), (\\alpha) is the learning rate, (\\epsilon > 0).

| PARAMETER      | DESCRIPTION                                                                                           |
| -------------- | ----------------------------------------------------------------------------------------------------- |
| `params`       | iterable of parameters to optimize or dicts defining parameter groups **TYPE:** `Iterable[Parameter]` |
| `lr`           | learning rate **TYPE:** `float` **DEFAULT:** `0.001`                                                  |
| `betas`        | coefficients used for running averages **TYPE:** `tuple[float, float]` **DEFAULT:** `(0.9, 0.999)`    |
| `eps`          | term added to the denominator to improve numerical stability **TYPE:** `float` **DEFAULT:** `1e-08`   |
| `weight_decay` | weight decay (L2 penalty) **TYPE:** `float` **DEFAULT:** `0.0`                                        |
| `dof`          | degrees of freedom **TYPE:** \`float                                                                  |

Source code in `holocron/optim/tadam.py`

```python
def __init__(
    self,
    params: Iterable[torch.nn.Parameter],
    lr: float = 1e-3,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    amsgrad: bool = False,
    dof: float | None = None,
) -> None:
    if lr < 0.0:
        raise ValueError(f"Invalid learning rate: {lr}")
    if eps < 0.0:
        raise ValueError(f"Invalid epsilon value: {eps}")
    if not 0.0 <= betas[0] < 1.0:
        raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
    if not 0.0 <= betas[1] < 1.0:
        raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
    if not weight_decay >= 0.0:
        raise ValueError(f"Invalid weight_decay value: {weight_decay}")
    defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay, "amsgrad": amsgrad, "dof": dof}
    super().__init__(params, defaults)
```

### AdaBelief

Bases: `Adam`

Implements the AdaBelief optimizer from ["AdaBelief Optimizer: Adapting Stepsizes by the Belief in Observed Gradients"](https://arxiv.org/pdf/2010.07468.pdf).

The estimation of momentums is described as follows, (\\forall t \\geq 1):

[ m_t \\leftarrow \\beta_1 m\_{t-1} + (1 - \\beta_1) g_t \\ s_t \\leftarrow \\beta_2 s\_{t-1} + (1 - \\beta_2) (g_t - m_t)^2 + \\epsilon ]

where (g_t) is the gradient of (\\theta_t), (\\beta_1, \\beta_2 \\in [0, 1]^2) are the exponential average smoothing coefficients, (m_0 = 0,\\ s_0 = 0), (\\epsilon > 0).

Then we correct their biases using:

[ \\hat{m_t} \\leftarrow \\frac{m_t}{1 - \\beta_1^t} \\ \\hat{s_t} \\leftarrow \\frac{s_t}{1 - \\beta_2^t} ]

And finally the update step is performed using the following rule:

[ \\theta_t \\leftarrow \\theta\_{t-1} - \\alpha \\frac{\\hat{m_t}}{\\sqrt{\\hat{s_t}} + \\epsilon} ]

where (\\theta_t) is the parameter value at step (t) ((\\theta_0) being the initialization value), (\\alpha) is the learning rate, (\\epsilon > 0).

| PARAMETER      | DESCRIPTION                                                           |
| -------------- | --------------------------------------------------------------------- |
| `params`       | iterable of parameters to optimize or dicts defining parameter groups |
| `lr`           | learning rate                                                         |
| `betas`        | coefficients used for running averages                                |
| `eps`          | term added to the denominator to improve numerical stability          |
| `weight_decay` | weight decay (L2 penalty)                                             |
| `amsgrad`      | whether to use the AMSGrad variant                                    |

### AdamP

```python
AdamP(params: Iterable[Parameter], lr: float = 0.001, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-08, weight_decay: float = 0.0, amsgrad: bool = False, delta: float = 0.1, mps_fused: bool = False)
```

Bases: `Adam`

Implements the AdamP optimizer from ["AdamP: Slowing Down the Slowdown for Momentum Optimizers on Scale-invariant Weights"](https://arxiv.org/pdf/2006.08217.pdf).

The estimation of momentums is described as follows, (\\forall t \\geq 1):

[ m_t \\leftarrow \\beta_1 m\_{t-1} + (1 - \\beta_1) g_t \\ v_t \\leftarrow \\beta_2 v\_{t-1} + (1 - \\beta_2) g_t^2 ]

where (g_t) is the gradient of (\\theta_t), (\\beta_1, \\beta_2 \\in [0, 1]^2) are the exponential average smoothing coefficients, (m_0 = g_0,\\ v_0 = 0).

Then we correct their biases using:

[ \\hat{m_t} \\leftarrow \\frac{m_t}{1 - \\beta_1^t} \\ \\hat{v_t} \\leftarrow \\frac{v_t}{1 - \\beta_2^t} ]

And finally the update step is performed using the following rule:

[ p_t \\leftarrow \\frac{\\hat{m_t}}{\\sqrt{\\hat{n_t} + \\epsilon}} \\ q_t \\leftarrow \\begin{cases} \\prod\_{\\theta_t}(p_t) & if\\ cos(\\theta_t, g_t) < \\delta / \\sqrt{dim(\\theta)}\\ p_t & \\text{otherwise}\\ \\end{cases} \\ \\theta_t \\leftarrow \\theta\_{t-1} - \\alpha q_t ]

where (\\theta_t) is the parameter value at step (t) ((\\theta_0) being the initialization value), (\\prod\_{\\theta_t}(p_t)) is the projection of (p_t) onto the tangent space of (\\theta_t), (cos(\\theta_t, g_t)) is the cosine similarity between (\\theta_t) and (g_t), (\\alpha) is the learning rate, (\\delta > 0), (\\epsilon > 0).

| PARAMETER      | DESCRIPTION                                                                                                                                                                       |
| -------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `params`       | iterable of parameters to optimize or dicts defining parameter groups **TYPE:** `Iterable[Parameter]`                                                                             |
| `lr`           | learning rate **TYPE:** `float` **DEFAULT:** `0.001`                                                                                                                              |
| `betas`        | coefficients used for running averages **TYPE:** `tuple[float, float]` **DEFAULT:** `(0.9, 0.999)`                                                                                |
| `eps`          | term added to the denominator to improve numerical stability **TYPE:** `float` **DEFAULT:** `1e-08`                                                                               |
| `weight_decay` | weight decay (L2 penalty) **TYPE:** `float` **DEFAULT:** `0.0`                                                                                                                    |
| `amsgrad`      | whether to use the AMSGrad variant **TYPE:** `bool` **DEFAULT:** `False`                                                                                                          |
| `delta`        | delta threshold for projection **TYPE:** `float` **DEFAULT:** `0.1`                                                                                                               |
| `mps_fused`    | opt into Metal fusion for contiguous FP32 MPS tensors without weight decay or AMSGrad; other configurations use the ordinary implementation **TYPE:** `bool` **DEFAULT:** `False` |

Source code in `holocron/optim/adamp.py`

```python
def __init__(
    self,
    params: Iterable[torch.nn.Parameter],
    lr: float = 1e-3,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    amsgrad: bool = False,
    delta: float = 0.1,
    mps_fused: bool = False,
) -> None:
    super().__init__(params, lr, betas, eps, weight_decay, amsgrad)
    self.delta = delta
    self.mps_fused = mps_fused
```

AdamP offers optional Metal kernel fusion on Apple Silicon:

```python
optimizer = holocron.optim.AdamP(model.parameters(), mps_fused=True)
```

A parameter group uses fusion only with contiguous FP32 MPS parameters, gradients, and state, zero weight decay, and `amsgrad=False`. Each parameter, its gradient, and its moment buffers must use distinct storage. Unsupported groups, missing `torch.mps.compile_shader`, and compilation failures use ordinary AdamP. Execution errors propagate without retrying and can leave parameters and moments partly updated. Restore both the model and optimizer from a checkpoint before continuing.

`mps_fused` defaults to `False` and remains a constructor preference when loading optimizer state. Mixed precision can still use fusion when parameters and unscaled gradients are FP32. Full-training accuracy equivalence remains unverified.

### Adan

```python
Adan(params: Iterable[Parameter], lr: float = 0.001, betas: tuple[float, float, float] = (0.98, 0.92, 0.99), eps: float = 1e-08, weight_decay: float = 0.0, amsgrad: bool = False)
```

Bases: `Optimizer`

Implements the Adan optimizer from ["Adan: Adaptive Nesterov Momentum Algorithm for Faster Optimizing Deep Models"](https://arxiv.org/pdf/2208.06677.pdf).

The estimation of momentums is described as follows, (\\forall t \\geq 1):

\[ m_t \\leftarrow \\beta_1 m\_{t-1} + (1 - \\beta_1) g_t \\ v_t \\leftarrow \\beta_2 v\_{t-1} + (1 - \\beta_2) (g_t - g\_{t-1}) \\ n_t \\leftarrow \\beta_3 n\_{t-1} + (1 - \\beta_3) [g_t + \\beta_2 (g_t - g\_{t - 1})]^2 \]

where (g_t) is the gradient of (\\theta_t), (\\beta_1, \\beta_2, \\beta_3 \\in [0, 1]^3) are the exponential average smoothing coefficients, (m_0 = g_0,\\ v_0 = 0,\\ n_0 = g_0^2).

Then we correct their biases using:

[ \\hat{m_t} \\leftarrow \\frac{m_t}{1 - \\beta_1^t} \\ \\hat{v_t} \\leftarrow \\frac{v_t}{1 - \\beta_2^t} \\ \\hat{n_t} \\leftarrow \\frac{n_t}{1 - \\beta_3^t} ]

And finally the update step is performed using the following rule:

[ p_t \\leftarrow \\frac{\\hat{m_t} + (1 - \\beta_2) \\hat{v_t}}{\\sqrt{\\hat{n_t} + \\epsilon}} \\ \\theta_t \\leftarrow \\frac{\\theta\_{t-1} - \\alpha p_t}{1 + \\lambda \\alpha} ]

where (\\theta_t) is the parameter value at step (t) ((\\theta_0) being the initialization value), (\\alpha) is the learning rate, (\\lambda \\geq 0) is the weight decay, (\\epsilon > 0).

| PARAMETER      | DESCRIPTION                                                                                                     |
| -------------- | --------------------------------------------------------------------------------------------------------------- |
| `params`       | iterable of parameters to optimize or dicts defining parameter groups **TYPE:** `Iterable[Parameter]`           |
| `lr`           | learning rate **TYPE:** `float` **DEFAULT:** `0.001`                                                            |
| `betas`        | coefficients used for running averages **TYPE:** `tuple[float, float, float]` **DEFAULT:** `(0.98, 0.92, 0.99)` |
| `eps`          | term added to the denominator to improve numerical stability **TYPE:** `float` **DEFAULT:** `1e-08`             |
| `weight_decay` | weight decay (L2 penalty) **TYPE:** `float` **DEFAULT:** `0.0`                                                  |
| `amsgrad`      | whether to use the AMSGrad variant **TYPE:** `bool` **DEFAULT:** `False`                                        |

Source code in `holocron/optim/adan.py`

```python
def __init__(
    self,
    params: Iterable[torch.nn.Parameter],
    lr: float = 1e-3,
    betas: tuple[float, float, float] = (0.98, 0.92, 0.99),
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    amsgrad: bool = False,
) -> None:
    if not lr >= 0.0:
        raise ValueError(f"Invalid learning rate: {lr}")
    if not eps >= 0.0:
        raise ValueError(f"Invalid epsilon value: {eps}")
    if len(betas) != 3:
        raise ValueError("Adan requires three beta parameters")
    for idx, beta in enumerate(betas):
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"Invalid beta parameter at index {idx}: {beta}")
    if not weight_decay >= 0.0:
        raise ValueError(f"Invalid weight_decay value: {weight_decay}")
    defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay, "amsgrad": amsgrad}
    super().__init__(params, defaults)
```

### AdEMAMix

```python
AdEMAMix(params: Iterable[Parameter], lr: float = 0.001, betas: tuple[float, float, float] = (0.9, 0.999, 0.9999), alpha: float = 5.0, eps: float = 1e-08, weight_decay: float = 0.0)
```

Bases: `Optimizer`

Implements the AdEMAMix optimizer from ["The AdEMAMix Optimizer: Better, Faster, Older"](https://arxiv.org/pdf/2409.03137).

The estimation of momentums is described as follows, (\\forall t \\geq 1):

[ m\_{1,t} \\leftarrow \\beta_1 m\_{1, t-1} + (1 - \\beta_1) g_t \\ m\_{2,t} \\leftarrow \\beta_3 m\_{2, t-1} + (1 - \\beta_3) g_t \\ s_t \\leftarrow \\beta_2 s\_{t-1} + (1 - \\beta_2) (g_t - m_t)^2 + \\epsilon ]

where (g_t) is the gradient of (\\theta_t), (\\beta_1, \\beta_2, \\beta_3 \\in [0, 1]^3) are the exponential average smoothing coefficients, (m\_{1,0} = 0,\\ m\_{2,0} = 0,\\ s_0 = 0), (\\epsilon > 0).

Then we correct their biases using:

[ \\hat{m\_{1,t}} \\leftarrow \\frac{m\_{1,t}}{1 - \\beta_1^t} \\ \\hat{s_t} \\leftarrow \\frac{s_t}{1 - \\beta_2^t} ]

And finally the update step is performed using the following rule:

[ \\theta_t \\leftarrow \\theta\_{t-1} - \\eta \\frac{\\hat{m\_{1,t}} + \\alpha m\_{2,t}}{\\sqrt{\\hat{s_t}} + \\epsilon} ]

where (\\theta_t) is the parameter value at step (t) ((\\theta_0) being the initialization value), (\\eta) is the learning rate, (\\alpha > 0) (\\epsilon > 0).

| PARAMETER      | DESCRIPTION                                                                                                       |
| -------------- | ----------------------------------------------------------------------------------------------------------------- |
| `params`       | iterable of parameters to optimize or dicts defining parameter groups **TYPE:** `Iterable[Parameter]`             |
| `lr`           | learning rate **TYPE:** `float` **DEFAULT:** `0.001`                                                              |
| `betas`        | coefficients used for running averages **TYPE:** `tuple[float, float, float]` **DEFAULT:** `(0.9, 0.999, 0.9999)` |
| `alpha`        | the exponential decay rate of the second moment estimates **TYPE:** `float` **DEFAULT:** `5.0`                    |
| `eps`          | term added to the denominator to improve numerical stability **TYPE:** `float` **DEFAULT:** `1e-08`               |
| `weight_decay` | weight decay (L2 penalty) **TYPE:** `float` **DEFAULT:** `0.0`                                                    |

Source code in `holocron/optim/ademamix.py`

```python
def __init__(
    self,
    params: Iterable[torch.nn.Parameter],
    lr: float = 1e-3,
    betas: tuple[float, float, float] = (0.9, 0.999, 0.9999),
    alpha: float = 5.0,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
) -> None:
    if lr < 0.0:
        raise ValueError(f"Invalid learning rate: {lr}")
    if eps < 0.0:
        raise ValueError(f"Invalid epsilon value: {eps}")
    for idx, beta in enumerate(betas):
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"Invalid beta parameter at index {idx}: {beta}")
    defaults = {"lr": lr, "betas": betas, "alpha": alpha, "eps": eps, "weight_decay": weight_decay}
    super().__init__(params, defaults)
```

## Optimizer wrappers

`holocron.optim` also implements optimizer wrappers.

A base optimizer should always be passed to the wrapper; e.g., you should write your code this way:

```python
optimizer = ...
optimizer = wrapper(optimizer)
```

### Lookahead

```python
Lookahead(base_optimizer: Optimizer, sync_rate: float = 0.5, sync_period: int = 6)
```

Bases: `Optimizer`

Implements the Lookahead optimizer wrapper from ["Lookahead Optimizer: k steps forward, 1 step back"](https://arxiv.org/pdf/1907.08610.pdf). <https://arxiv.org/pdf/1907.08610.pdf>\`\_.

Example

```python
from torch.optim import AdamW
from holocron.optim.wrapper import Lookahead
model = ...
opt = AdamW(model.parameters(), lr=3e-4)
opt_wrapper = Lookahead(opt)
```

| PARAMETER        | DESCRIPTION                                                                                             |
| ---------------- | ------------------------------------------------------------------------------------------------------- |
| `base_optimizer` | base parameter optimizer **TYPE:** `Optimizer`                                                          |
| `sync_rate`      | rate of weight synchronization **TYPE:** `float` **DEFAULT:** `0.5`                                     |
| `sync_period`    | number of step performed on fast weights before weight synchronization **TYPE:** `int` **DEFAULT:** `6` |

Source code in `holocron/optim/wrapper.py`

```python
def __init__(
    self,
    base_optimizer: torch.optim.Optimizer,
    sync_rate: float = 0.5,
    sync_period: int = 6,
) -> None:
    if sync_rate < 0 or sync_rate > 1:
        raise ValueError(f"expected positive float lower than 1 as sync_rate, received: {sync_rate}")
    if not isinstance(sync_period, int) or sync_period < 1:
        raise ValueError(f"expected positive integer as sync_period, received: {sync_period}")
    # Optimizer attributes
    self.defaults = {"sync_rate": sync_rate, "sync_period": sync_period}
    self.state = defaultdict(dict)
    # Base optimizer attributes
    self.base_optimizer = base_optimizer
    # Wrapper attributes
    self.fast_steps = 0
    self.param_groups = []
    for group in self.base_optimizer.param_groups:
        self._add_param_group(group)
```

### Scout

```python
Scout(base_optimizer: Optimizer, sync_rate: float = 0.5, sync_period: int = 6)
```

Bases: `Lookahead`

Implements a new optimizer wrapper based on ["Lookahead Optimizer: k steps forward, 1 step back"](https://arxiv.org/pdf/1907.08610.pdf).

Example

```python
from torch.optim import AdamW
from holocron.optim.wrapper import Scout
model = ...
opt = AdamW(model.parameters(), lr=3e-4)
opt_wrapper = Scout(opt)
```

| PARAMETER        | DESCRIPTION                                                                                             |
| ---------------- | ------------------------------------------------------------------------------------------------------- |
| `base_optimizer` | base parameter optimizer **TYPE:** `Optimizer`                                                          |
| `sync_rate`      | rate of weight synchronization **TYPE:** `float` **DEFAULT:** `0.5`                                     |
| `sync_period`    | number of step performed on fast weights before weight synchronization **TYPE:** `int` **DEFAULT:** `6` |

Source code in `holocron/optim/wrapper.py`

```python
def __init__(
    self,
    base_optimizer: torch.optim.Optimizer,
    sync_rate: float = 0.5,
    sync_period: int = 6,
) -> None:
    super().__init__(base_optimizer, sync_rate, sync_period)
    self.buffer = [p.data.unsqueeze(0) for group in self.param_groups for p in group["params"]]
```
