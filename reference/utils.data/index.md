# holocron.utils.data

## Batch collate

### Mixup

```python
Mixup(num_classes: int, alpha: float = 0.2)
```

Bases: `Module`

Implements a batch collate function with MixUp strategy from ["mixup: Beyond Empirical Risk Minimization"](https://arxiv.org/pdf/1710.09412.pdf).

Example

```python
import torch
from holocron.utils.data import Mixup
mix = Mixup(num_classes=10, alpha=0.4)
loader = torch.utils.data.DataLoader(dataset, batch_size, collate_fn=mix.collate)
```

| PARAMETER     | DESCRIPTION                                       |
| ------------- | ------------------------------------------------- |
| `num_classes` | number of expected classes **TYPE:** `int`        |
| `alpha`       | mixup factor **TYPE:** `float` **DEFAULT:** `0.2` |

Source code in `holocron/utils/data/collate.py`

```python
def __init__(self, num_classes: int, alpha: float = 0.2) -> None:
    super().__init__()
    self.num_classes: int = num_classes
    if alpha < 0:
        raise ValueError("`alpha` only takes positive values")
    self.alpha: float = alpha
```

#### collate

```python
collate(batch: list[tuple[Tensor, int]]) -> tuple[Tensor, Tensor]
```

Collate and mix a batch using a worker-picklable bound method.

| PARAMETER | DESCRIPTION                                                           |
| --------- | --------------------------------------------------------------------- |
| `batch`   | dataset images and class indices **TYPE:** `list[tuple[Tensor, int]]` |

| RETURNS                 | DESCRIPTION                                   |
| ----------------------- | --------------------------------------------- |
| `tuple[Tensor, Tensor]` | Collated mixed images and soft class targets. |

Source code in `holocron/utils/data/collate.py`

```python
def collate(self, batch: list[tuple[Tensor, int]]) -> tuple[Tensor, Tensor]:
    """Collate and mix a batch using a worker-picklable bound method.

    Args:
        batch: dataset images and class indices

    Returns:
        Collated mixed images and soft class targets.
    """
    return self(*default_collate(batch))
```
