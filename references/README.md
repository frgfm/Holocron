# Holocron training scripts

This section is specific to train computer vision models.


## Installation

### Prerequisites

Python 3.6 (or higher) and [pip](https://pip.pypa.io/en/stable/) & [Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) are required to install Holocron.


### Developer mode

In order to install the specific dependencies for training, you will have to install the package from source *(install [Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) first)*:

```shell
git clone https://github.com/frgfm/Holocron.git
pip install -e "Holocron/.[training]"
```

## Available tasks

### Image classification

Refer to the [`./classification`](classification) folder

### Semantic segmentation

Refer to the [`./segmentation`](segmentation) folder

### Object detection

Refer to the [`./detection`](detection) folder
