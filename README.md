# Holocron

Implementations of recent Deep Learning tricks in Computer Vision, easily paired up with your favorite framework and model zoo.

> **Holocrons** were information-storage [datacron](https://starwars.fandom.com/wiki/Datacron) devices used by both the [Jedi Order](https://starwars.fandom.com/wiki/Jedi_Order) and the [Sith](https://starwars.fandom.com/wiki/Sith) that contained ancient lessons or valuable information in [holographic](https://starwars.fandom.com/wiki/Hologram) form.

*Source: Wookieepedia*

## Installation

This package was developed using minimal dependencies ([pytorch](https://github.com/pytorch/pytorch), [torchvision](https://github.com/pytorch/vision)). You can install it using the following commands:

```bash
git clone https://github.com/frgfm/Holocron.git
pip install -e Holocron/
```

## Usage

### models

Using the models module, you can easily load torch modules or full models:

```python
from holocron.models.resnets import TridentBlock
# Load pretrained Resnet
model = TridentBlock(64, 16, branches=3)
model.eval()
```

Then, let's generate a random feature maps

```python
import torch
# Get random inputs
x1 = torch.rand(1, 64, 256, 256)
x2 = torch.rand(1, 64, 256, 256)
x3 = torch.rand(1, 64, 256, 256)
```

Now we can move them to GPU and forward them

```python
# Move inputs and model to GPU
if torch.cuda.is_available():
    model = model.cuda()
    x1, x2, x3 = x1.cuda(), x2.cuda(), x3.cuda()
# Forward
with torch.no_grad():
    output = model([x1, x2, x3])
```



### optim.lr_scheduler

You can use the OneCycleLR scheduler as follows:

```python
from torchvision.models.resnet import resnet50
from torch.optim import Adam
from holocron.optim.lr_scheduler import OneCycleScheduler

model = resnet50()
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



## Submitting a request / Reporting an issue

Regarding issues, use the following format for the title:

> [Topic] Your Issue name

Example:

> [models resnet] Add spectral normalization option