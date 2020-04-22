# Holocron

[![License](https://img.shields.io/badge/License-MIT-brightgreen.svg)](LICENSE) [![Codacy Badge](https://api.codacy.com/project/badge/Grade/5713eafaf8074e27a4013dbfcfad9d69)](https://www.codacy.com/manual/fg/Holocron?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=frgfm/Holocron&amp;utm_campaign=Badge_Grade) ![Build Status](https://github.com/frgfm/Holocron/workflows/python-package/badge.svg) [![codecov](https://codecov.io/gh/frgfm/Holocron/branch/master/graph/badge.svg)](https://codecov.io/gh/frgfm/Holocron) [![Docs](https://img.shields.io/badge/docs-available-blue.svg)](https://frgfm.github.io/Holocron)

Implementations of recent Deep Learning tricks in Computer Vision, easily paired up with your favorite framework and model zoo.

> **Holocrons** were information-storage [datacron](https://starwars.fandom.com/wiki/Datacron) devices used by both the [Jedi Order](https://starwars.fandom.com/wiki/Jedi_Order) and the [Sith](https://starwars.fandom.com/wiki/Sith) that contained ancient lessons or valuable information in [holographic](https://starwars.fandom.com/wiki/Hologram) form.

*Source: Wookieepedia*



## Table of Contents

- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Technical Roadmap](#technical-roadmap)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)



*Note: support of activation mapper and model summary has been dropped and outsourced to independent packages ([torch-cam](https://github.com/frgfm/torch-cam) & [torch-scan](https://github.com/frgfm/torch-scan)) to clarify project scope.*



## Getting started

### Prerequisites

- Python 3.6 (or more recent)
- [pip](https://pip.pypa.io/en/stable/)

### Installation

Install the package in developer mode

```bash
git clone https://github.com/frgfm/Holocron.git
pip install -e Holocron/
```

*Note: pip package release will soon be available*



## Usage

### nn

##### Main features

- Activation: [Mish](https://arxiv.org/abs/1908.08681), [NLReLU](https://arxiv.org/abs/1908.03682)
- Loss: [Focal Loss](https://arxiv.org/abs/1708.02002)

##### Usage

Similar usage to  `torch.nn`

```python
import torch.nn as nn
from holocron.nn import Mish, NLReLU

# Both modules inherit from torch.nn.Module and can be used as such
model = nn.Sequential(nn.Conv2d(3, 64, (3, 3)),
                      Mish(),
                      nn.Conv2d(64, 128, (3, 3)),
                      NLReLU(),)
```



### models

##### Main features

- Classification: [Res2Net](https://arxiv.org/abs/1904.01169), based on the great [implementation](https://github.com/gasvn/Res2Net) from gasvn

##### Usage

Using the `models` module, you can easily load torch modules or full models:

```python
from holocron.models.res2net import res2net
# Load pretrained Res2net
model = res2net(depth=50, num_classes=10, pretrained=True).eval()
```



### ops

##### Main features

- boxes: [Distance-IoU & Complete-IoU losses](https://arxiv.org/abs/1911.08287)

##### Usage

Similar usage to `torchvision.ops`

```python
import torch
from holocron.ops.boxes import box_ciou

boxes1 = torch.tensor([[0, 0, 100, 100], [100, 100, 200, 200]], dtype=torch.float32)
boxes1 = torch.tensor([[50, 50, 150, 150]], dtype=torch.float32)

box_ciou(boxes1, boxes2)
```



### optim

##### Main features

- Optimizer: [LARS](https://arxiv.org/abs/1708.03888), [Lamb](https://arxiv.org/abs/1904.00962), [RAdam](https://arxiv.org/abs/1908.03265) and customized versions (RaLars)
- Optimizer wrapper: [Lookahead](https://arxiv.org/abs/1907.08610), Scout (experimental)
- Scheduler: [OneCycleScheduler](https://arxiv.org/abs/1803.09820)

##### Usage

The optimizer wrapper can be used on any `torch.optim.optimizer.Optimizer` object 

```python
from torchvision.models.resnet import resnet18
from holocron.optim import RaLars

model = resnet18()
# Common usage of optimizer
optimizer = RaLars(model.parameters(), lr=3e-4)
# Wrap it with Lookahead
optimizer = Lookahead(optimizer, sync_rate=0.5, sync_period=6)
# Now use it just like your base optimizer
```



You can use the `OneCycleScheduler` as follows:

```python
from torchvision.models.resnet import resnet18
from torch.optim import Adam
from holocron.optim.lr_scheduler import OneCycleScheduler

model = resnet18()
# Let's have different LRs for weight and biases for instance
bias_params, weight_params = [], []
for n, p in model.named_parameters():
	if n.endswith('.bias'):
		bias_params.append(p)
    else:
    	weight_params.append(p)
# We pass the parameters to the optimizer
optimizer = Adam([dict(params=weight_params, lr=2e-4), dict(params=bias_params, lr=1e-4)])

steps = 500
scheduler = OneCycleScheduler(optimizer, steps, cycle_momentum=False)
# Let's record the evolution of LR in each group
lrs = [[], []]
for step in range(steps):
	for idx, group in enumerate(optimizer.param_groups):
		lrs[idx].append(group['lr'])
	# Train your model and perform optimizer.step() here
	scheduler.step()

# And plot the result
import matplotlib.pyplot as plt
plt.plot(lrs[0], label='Weight LR'); plt.plot(lrs[1], label='Bias LR'); plt.legend(); plt.show()
```

![onecycle](static/images/onecycle.png)



## Technical roadmap

The project is currently under development, here are the objectives for the next releases:

- [ ] Standardize models: standardize models by task.
- [ ] Speed benchmark: compare `holocron.nn` functions execution speed.
- [ ] Reference scripts: add reference training scripts



## Documentation

The full package documentation is available [here](<https://frgfm.github.io/Holocron/>) for detailed specifications. The documentation was built with [Sphinx](sphinx-doc.org) using a [theme](github.com/readthedocs/sphinx_rtd_theme) provided by [Read the Docs](readthedocs.org) 



## Contributing

Please refer to `CONTRIBUTING` if you wish to contribute to this project.



## License

Distributed under the MIT License. See `LICENSE` for more information.