# Copyright (C) 2022-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

from enum import StrEnum
from math import sqrt
from typing import Any

import torch
from PIL import Image
from torch import nn
from torchvision.transforms import v2 as T
from torchvision.transforms.v2.functional import pad, resize

__all__ = ["RandomZoomOut", "Resize", "ResizeMethod"]


class ResizeMethod(StrEnum):
    """Resize methods
    Available methods are ``squish``, ``pad``.
    """

    SQUISH = "squish"
    PAD = "pad"


def _get_image_shape(image: Image.Image | torch.Tensor) -> tuple[int, int]:
    if isinstance(image, torch.Tensor):
        if image.ndim != 3:
            raise ValueError("the input tensor is expected to be 3-dimensional")
        h, w = image.shape[1:]
    elif isinstance(image, Image.Image):
        w, h = image.size
    else:
        raise TypeError("expected arg 'image' to be a PIL image or a torch.Tensor")

    return h, w


class Resize(T.Resize):
    """Implements a more flexible resizing scheme.

    ![Resize example](https://github.com/frgfm/Holocron/releases/download/v0.2.1/resize_example.png)

    Example:
        ```python
        from holocron.transforms import Resize, ResizeMethod
        pil_img = ...
        tf = Resize((224, 224), mode=ResizeMethod.PAD)
        resized_img = tf(pil_img)
        ```

    Args:
        size: the desired height and width of the image in pixels
        mode: the resizing scheme ("squish" is similar to PyTorch, "pad" will preserve the aspect ratio and pad)
        pad_mode: padding mode when `mode` is "pad"
        kwargs: the keyword arguments of [`torchvision.transforms.v2.Resize`][torchvision.transforms.v2.Resize]
    """

    def __init__(
        self,
        size: tuple[int, int],
        mode: ResizeMethod = ResizeMethod.SQUISH,
        pad_mode: str = "constant",
        **kwargs: Any,
    ) -> None:
        if not isinstance(mode, ResizeMethod):
            raise TypeError("mode is expected to be a ResizeMethod")
        if not isinstance(size, (tuple, list)) or len(size) != 2 or any(s <= 0 for s in size):
            raise ValueError("size is expected to be a sequence of 2 positive integers")
        super().__init__(size, **kwargs)
        self.mode: ResizeMethod = mode
        self.pad_mode: str = pad_mode
        self.size: tuple[int, int]

    def get_params(self, image: Image.Image | torch.Tensor) -> tuple[int, int]:  # noqa: D102
        h, w = _get_image_shape(image)
        o_ratio = h / w
        if self.size[0] / self.size[1] > o_ratio:
            h_, w_ = round(self.size[1] * o_ratio), self.size[1]
        else:
            h_, w_ = self.size[0], round(self.size[0] / o_ratio)

        return h_, w_

    def forward(self, image: Image.Image | torch.Tensor) -> Image.Image | torch.Tensor:  # noqa: D102
        if self.mode == ResizeMethod.SQUISH:
            return super().forward(image)
        h, w = self.get_params(image)
        img = resize(image, (h, w), self.interpolation)  # ty: ignore[invalid-argument-type]
        # get the padding
        h_pad, w_pad = self.size[0] - h, self.size[1] - w
        padding = w_pad // 2, h_pad // 2, w_pad - w_pad // 2, h_pad - h_pad // 2
        # Fill the rest up to target_size
        return pad(img, padding, padding_mode=self.pad_mode)  # ty: ignore[invalid-argument-type]


class RandomZoomOut(nn.Module):
    """Implements a size reduction of the orignal image to provide a zoom out effect.

    ![RandomZoomOut example](https://github.com/frgfm/Holocron/releases/download/v0.2.1/randomzoomout_example.png)

    Example:
        ```python
        from holocron.transforms import RandomZoomOut
        pil_img = ...
        tf = RandomZoomOut((224, 224), scale=(0.3, 1.))
        resized_img = tf(pil_img)
        ```

    Args:
        size: the desired height and width of the image in pixels
        scale: the range of relative area of the projected image to the desired size
        kwargs: the keyword arguments of [`torchvision.transforms.functional.resize`][torchvision.transforms.functional.resize]
    """

    def __init__(self, size: tuple[int, int], scale: tuple[float, float] = (0.5, 1.0), **kwargs: Any) -> None:
        if not isinstance(size, (tuple, list)) or len(size) != 2 or any(s <= 0 for s in size):
            raise ValueError("size is expected to be a sequence of 2 positive integers")
        if len(scale) != 2 or scale[0] > scale[1]:
            raise ValueError("scale is expected to be a couple of floats, the first one being small than the second")
        super().__init__()
        self.size: tuple[int, int] = size
        self.scale: tuple[float, float] = scale
        self._kwargs: dict[str, Any] = kwargs

    def get_params(self, image: Image.Image | torch.Tensor) -> tuple[int, int]:  # noqa: D102
        h, w = _get_image_shape(image)

        scale = (self.scale[1] - self.scale[0]) * torch.rand(1).item() + self.scale[0]
        aratio = h / w
        # Preserve the aspect ratio
        tratio = self.size[0] / self.size[1]
        max_area = self.size[1] ** 2 * aratio if tratio > aratio else self.size[0] ** 2 / aratio
        area = max_area * scale

        w_ = round(sqrt(area / aratio))
        h_ = round(area / w_)

        return h_, w_

    def forward(self, image: Image.Image | torch.Tensor) -> Image.Image | torch.Tensor:  # noqa: D102  # ty: ignore[invalid-mutable-override]
        # Skip dummy cases
        if self.scale[0] == 1:
            return image
        # Get the size of the small image
        h, w = self.get_params(image)
        # Resize the image to this
        img = resize(image, (h, w), **self._kwargs)  # ty: ignore[invalid-argument-type]
        # get the padding
        h_delta, w_delta = self.size[0] - h, self.size[1] - w
        padding = w_delta // 2, h_delta // 2, w_delta - w_delta // 2, h_delta - h_delta // 2
        # Fill the rest up to size
        return pad(img, padding)  # ty: ignore[invalid-argument-type]
