# Copyright (C) 2022, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

from enum import Enum
from math import sqrt
from typing import Any, Tuple, Union

import torch
from PIL import Image
from torch import nn
from torchvision.transforms import transforms as T
from torchvision.transforms.functional import pad, resize

__all__ = ["Resize", "RandomZoomOut"]


class ResizeMethod(Enum):
    """Resize methods
    Available methods are ``squish``, ``pad``.
    """

    SQUISH = "squish"
    PAD = "pad"


def _get_image_shape(image: Union[Image.Image, torch.Tensor]) -> Tuple[int, int]:
    if isinstance(image, torch.Tensor):
        assert image.ndim == 3
        h, w = image.shape[1:]
    elif isinstance(image, Image.Image):
        w, h = image.size
    else:
        raise TypeError("expected arg 'image' to be a PIL image or a torch.Tensor")

    return h, w


class Resize(T.Resize):
    """Implements a more flexible resizing scheme.

    .. image:: https://github.com/frgfm/Holocron/releases/download/v0.2.1/resize_example.png
        :align: center

    >>> from holocron.transforms import Resize
    >>> pil_img = ...
    >>> tf = Resize((224, 224), mode="pad")
    >>> resized_img = tf(pil_img)

    Args:
        size: the desired height and width of the image in pixels
        mode: the resizing scheme ("squish" is similar to PyTorch, "pad" will preserve the aspect ratio and pad)
        pad_mode: padding mode when `mode` is "pad"
        kwargs: the keyword arguments of `torchvision.transforms.Resize`

    Returns:
        the resized image
    """

    def __init__(
        self,
        size: Tuple[int, int],
        mode: ResizeMethod = ResizeMethod.SQUISH,
        pad_mode: str = "constant",
        **kwargs: Any,
    ) -> None:
        assert isinstance(mode, ResizeMethod)
        assert isinstance(size, (tuple, list)) and len(size) == 2 and all(s > 0 for s in size)
        super().__init__(size, **kwargs)
        self.mode = mode
        self.pad_mode = pad_mode

    def get_params(self, image: Union[Image.Image, torch.Tensor]) -> Tuple[int, int]:
        h, w = _get_image_shape(image)
        o_ratio = h / w
        if self.size[0] / self.size[1] > o_ratio:
            _h, _w = int(round(self.size[1] * o_ratio)), self.size[1]
        else:
            _h, _w = self.size[0], int(round(self.size[0] / o_ratio))

        return _h, _w

    def forward(self, image: Union[Image.Image, torch.Tensor]) -> Union[Image.Image, torch.Tensor]:
        if self.mode == ResizeMethod.SQUISH:
            return super().forward(image)
        else:
            h, w = self.get_params(image)
            img = resize(image, (h, w), self.interpolation)
            # get the padding
            h_pad, w_pad = self.size[0] - h, self.size[1] - w
            _padding = w_pad // 2, h_pad // 2, w_pad - w_pad // 2, h_pad - h_pad // 2
            # Fill the rest up to target_size
            return pad(img, _padding, padding_mode=self.pad_mode)


class RandomZoomOut(nn.Module):
    """Implements a size reduction of the orignal image to provide a zoom out effect.

    .. image:: https://github.com/frgfm/Holocron/releases/download/v0.2.1/randomzoomout_example.png
        :align: center

    >>> from holocron.transforms import RandomZoomOut
    >>> pil_img = ...
    >>> tf = RandomZoomOut((224, 224), scale=(0.3, 1.))
    >>> resized_img = tf(pil_img)

    Args:
        size: the desired height and width of the image in pixels
        scale: the range of relative area of the projected image to the desired size
        kwargs: the keyword arguments of `torchvision.transforms.functional.resize`

    Returns:
        the resized image
    """

    def __init__(self, size: Tuple[int, int], scale: Tuple[float, float] = (0.5, 1.0), **kwargs: Any):
        assert isinstance(size, (tuple, list)) and len(size) == 2 and all(s > 0 for s in size)
        assert len(scale) == 2 and scale[0] <= scale[1]
        super().__init__()
        self.size = size
        self.scale = scale
        self._kwargs = kwargs

    def get_params(self, image: Union[Image.Image, torch.Tensor]) -> Tuple[int, int]:
        h, w = _get_image_shape(image)

        _scale = (self.scale[1] - self.scale[0]) * torch.rand(1).item() + self.scale[0]
        _aratio = h / w
        # Preserve the aspect ratio
        _tratio = self.size[0] / self.size[1]
        if _tratio > _aratio:
            _max_area = self.size[1] ** 2 * _aratio
        else:
            _max_area = self.size[0] ** 2 / _aratio
        _area = _max_area * _scale

        _w = int(round(sqrt(_area / _aratio)))
        _h = int(round(_area / _w))

        return _h, _w

    def forward(self, image: Union[Image.Image, torch.Tensor]) -> Union[Image.Image, torch.Tensor]:
        # Skip dummy cases
        if self.scale[0] == 1:
            return image
        # Get the size of the small image
        h, w = self.get_params(image)
        # Resize the image to this
        img = resize(image, (h, w), **self._kwargs)
        # get the padding
        h_delta, w_delta = self.size[0] - h, self.size[1] - w
        _padding = w_delta // 2, h_delta // 2, w_delta - w_delta // 2, h_delta - h_delta // 2
        # Fill the rest up to size
        return pad(img, _padding)
