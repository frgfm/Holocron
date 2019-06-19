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



## Submitting a request / Reporting an issue

Regarding issues, use the following format for the title:

> [Topic] Your Issue name

Example:

> [models resnet] Add spectral normalization option