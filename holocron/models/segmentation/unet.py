import sys
import torch
from torch import Tensor
import torch.nn as nn
from typing import Dict, Any, Union, Optional, Callable, List

from ...nn.init import init_module
from ..utils import conv_sequence, load_pretrained_params


__all__ = ['UNet', 'unet', 'UNetp', 'unetp', 'UNetpp', 'unetpp', 'UNet3p', 'unet3p']


default_cfgs: Dict[str, Dict[str, Any]] = {
    'unet': {'arch': 'UNet',
             'layout': [64, 128, 256, 512, 1024],
             'url': None},
    'unetp': {'arch': 'UNetp',
              'layout': [64, 128, 256, 512, 1024],
              'url': None},
    'unetpp': {'arch': 'UNetpp',
               'layout': [64, 128, 256, 512, 1024],
               'url': None},
    'unet3p': {'arch': 'UNet3p',
               'layout': [64, 128, 256, 512, 1024],
               'url': None}
}


def down_path(
    in_chan: int,
    out_chan: int,
    downsample: bool = True,
    padding: int = 0,
    act_layer: Optional[nn.Module] = None,
    norm_layer: Optional[Callable[[int], nn.Module]] = None,
    drop_layer: Optional[Callable[..., nn.Module]] = None,
    conv_layer: Optional[Callable[..., nn.Module]] = None
) -> nn.Sequential:

    layers: List[nn.Module] = [nn.MaxPool2d(2)] if downsample else []
    layers.extend([*conv_sequence(in_chan, out_chan, act_layer, norm_layer, drop_layer, conv_layer,
                                  kernel_size=3, padding=padding),
                   *conv_sequence(out_chan, out_chan, act_layer, norm_layer, drop_layer, conv_layer,
                                  kernel_size=3, padding=padding)])
    return nn.Sequential(*layers)


class UpPath(nn.Module):
    def __init__(
        self,
        in_chan: int,
        out_chan: int,
        num_skips: int = 1,
        bilinear_upsampling: bool = True,
        padding: int = 0,
        act_layer: Optional[nn.Module] = None,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        drop_layer: Optional[Callable[..., nn.Module]] = None,
        conv_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super().__init__()

        self.upsample: nn.Module
        if bilinear_upsampling:
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.upsample = nn.ConvTranspose2d(in_chan, in_chan // 2, 2, stride=2)

        self.block = nn.Sequential(*conv_sequence(in_chan, out_chan,
                                                  act_layer, norm_layer, drop_layer, conv_layer,
                                                  kernel_size=3, padding=padding),
                                   *conv_sequence(out_chan, out_chan,
                                                  act_layer, norm_layer, drop_layer, conv_layer,
                                                  kernel_size=3, padding=padding))
        self.num_skips = num_skips

    def forward(self, downfeats: Union[Tensor, List[Tensor]], upfeat: Tensor) -> Tensor:

        if not isinstance(downfeats, list):
            downfeats = [downfeats]
        if len(downfeats) != self.num_skips:
            raise ValueError(f"Expected {self.num_skips} encoding feats, received {len(downfeats)}")
        # Upsample expansive features
        _upfeat = self.upsample(upfeat)
        # Crop contracting path features
        for idx, downfeat in enumerate(downfeats):
            if downfeat.shape != _upfeat.shape:
                delta_w = downfeat.shape[-1] - _upfeat.shape[-1]
                w_slice = slice(delta_w // 2, -delta_w // 2 if delta_w > 0 else downfeat.shape[-1])
                delta_h = downfeat.shape[-2] - _upfeat.shape[-2]
                h_slice = slice(delta_h // 2, -delta_h // 2 if delta_h > 0 else downfeat.shape[-2])
                downfeats[idx] = downfeat[..., h_slice, w_slice]
        # Concatenate both feature maps and forward them
        return self.block(torch.cat((*downfeats, _upfeat), dim=1))


class UNet(nn.Module):
    """Implements a U-Net architecture

    Args:
        layout: number of channels after each contracting block
        in_channels: number of channels in the input tensor
        num_classes: number of output classes
        act_layer: activation layer
        norm_layer: normalization layer
        drop_layer: dropout layer
        conv_layer: convolutional layer
        same_padding: enforces same padding in convolutions
        bilinear_upsampling: replaces transposed conv by bilinear interpolation for upsampling
    """
    def __init__(
        self,
        layout: List[int],
        in_channels: int = 3,
        num_classes: int = 10,
        act_layer: Optional[nn.Module] = None,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        drop_layer: Optional[Callable[..., nn.Module]] = None,
        conv_layer: Optional[Callable[..., nn.Module]] = None,
        same_padding: bool = True,
        bilinear_upsampling: bool = True,
    ) -> None:
        super().__init__()

        if act_layer is None:
            act_layer = nn.ReLU(inplace=True)

        # Contracting path
        self.encoders = nn.ModuleList([])
        _layout = [in_channels] + layout
        if bilinear_upsampling:
            _layout[-1] = _layout[-1] // 2
        _pool = False
        for in_chan, out_chan in zip(_layout[:-1], _layout[1:]):
            self.encoders.append(down_path(in_chan, out_chan, _pool, int(same_padding),
                                           act_layer, norm_layer, drop_layer, conv_layer))
            _pool = True

        # Expansive path
        self.decoders = nn.ModuleList([])
        _layout = [chan // 2 if bilinear_upsampling else chan for chan in layout[::-1][1:-1]] + [layout[0]]
        for in_chan, out_chan in zip(layout[::-1][:-1], _layout):
            self.decoders.append(UpPath(in_chan, out_chan, 1, bilinear_upsampling, int(same_padding),
                                        act_layer, norm_layer, drop_layer, conv_layer))

        # Classifier
        self.classifier = nn.Conv2d(layout[0], num_classes, 1)

        init_module(self, 'relu')

    def forward(self, x: Tensor) -> Tensor:

        xs: List[Tensor] = []
        # Contracting path
        for encoder in self.encoders[:-1]:
            xs.append(encoder(xs[-1] if len(xs) > 0 else x))
        x = self.encoders[-1](xs[-1])

        # Expansive path
        for decoder in self.decoders:
            x = decoder(xs.pop(), x)

        # Classifier
        x = self.classifier(x)
        return x


class UNetp(nn.Module):
    """Implements a UNet+ architecture

    Args:
        layout (list<int>): number of channels after each contracting block
        in_channels (int, optional): number of channels in the input tensor
        num_classes (int, optional): number of output classes
        act_layer: activation layer
        norm_layer: normalization layer
        drop_layer: dropout layer
        conv_layer: convolutional layer
    """
    def __init__(
        self,
        layout: List[int],
        in_channels: int = 3,
        num_classes: int = 10,
        act_layer: Optional[nn.Module] = None,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        drop_layer: Optional[Callable[..., nn.Module]] = None,
        conv_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()

        if act_layer is None:
            act_layer = nn.ReLU(inplace=True)

        # Contracting path
        self.encoders = nn.ModuleList([])
        _layout = [in_channels] + layout
        _pool = False
        for in_chan, out_chan in zip(_layout[:-1], _layout[1:]):
            self.encoders.append(down_path(in_chan, out_chan, _pool, 1,
                                           act_layer, norm_layer, drop_layer, conv_layer))
            _pool = True

        # Expansive path
        self.decoders = nn.ModuleList([])
        for next_chan, row_chan, num_cells in zip(layout[1:], layout[:-1], range(len(_layout) - 1, 0, -1)):
            self.decoders.append(nn.ModuleList([
                UpPath(next_chan + row_chan, row_chan, 1, True, 1,
                       act_layer, norm_layer, drop_layer, conv_layer)
                for _ in range(num_cells + 1)
            ]))

        # Classifier
        self.classifier = nn.Conv2d(layout[0], num_classes, 1)

        init_module(self, 'relu')

    def forward(self, x: Tensor) -> Tensor:

        xs: List[Tensor] = []
        # Contracting path
        for encoder in self.encoders:
            xs.append(encoder(xs[-1] if len(xs) > 0 else x))

        # Nested expansive path
        for j in range(len(self.decoders)):
            for i in range(len(self.decoders) - j):
                xs[i] = self.decoders[i][j](xs[i], xs[i + 1] if (i + 1) < (len(self.decoders) - j) else xs.pop())

        return self.classifier(xs.pop())


class UNetpp(nn.Module):
    """Implements a UNet++ architecture

    Args:
        layout (list<int>): number of channels after each contracting block
        in_channels (int, optional): number of channels in the input tensor
        num_classes (int, optional): number of output classes
        act_layer: activation layer
        norm_layer: normalization layer
        drop_layer: dropout layer
        conv_layer: convolutional layer
    """
    def __init__(
        self,
        layout: List[int],
        in_channels: int = 3,
        num_classes: int = 10,
        act_layer: Optional[nn.Module] = None,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        drop_layer: Optional[Callable[..., nn.Module]] = None,
        conv_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super().__init__()

        if act_layer is None:
            act_layer = nn.ReLU(inplace=True)

        # Contracting path
        self.encoders = nn.ModuleList([])
        _layout = [in_channels] + layout
        _pool = False
        for in_chan, out_chan in zip(_layout[:-1], _layout[1:]):
            self.encoders.append(down_path(in_chan, out_chan, _pool, 1, act_layer, norm_layer, drop_layer, conv_layer))
            _pool = True

        # Expansive path
        self.decoders = nn.ModuleList([])
        for next_chan, row_chan, num_cells in zip(layout[1:], layout[:-1], range(len(_layout) - 1, 0, -1)):
            self.decoders.append(nn.ModuleList([
                UpPath(next_chan + num_skips * row_chan, row_chan, num_skips, True, 1,
                       act_layer, norm_layer, drop_layer, conv_layer)
                for num_skips in range(1, num_cells + 2)
            ]))

        # Classifier
        self.classifier = nn.Conv2d(layout[0], num_classes, 1)

        init_module(self, 'relu')

    def forward(self, x: Tensor) -> Tensor:

        xs: List[List[Tensor]] = []
        # Contracting path
        for encoder in self.encoders:
            xs.append([encoder(xs[-1][0] if len(xs) > 0 else x)])

        # Nested expansive path
        for j in range(len(self.decoders)):
            for i in range(len(self.decoders) - j):
                xs[i].append(self.decoders[i][j](
                    xs[i][:j + 1],
                    xs[i + 1][j] if (i + 1) < (len(self.decoders) - j) else xs.pop()[-1]
                ))

        # Classifier
        x = self.classifier(xs.pop()[-1])
        return x


class FSAggreg(nn.Module):
    def __init__(
        self,
        e_chans: List[int],
        skip_chan: int,
        d_chans: List[int],
        act_layer: Optional[nn.Module] = None,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        drop_layer: Optional[Callable[..., nn.Module]] = None,
        conv_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:

        super().__init__()

        # Check stem conv channels
        base_chan = e_chans[0] if len(e_chans) > 0 else skip_chan
        # Get UNet depth
        depth = len(e_chans) + 1 + len(d_chans)
        # Downsample = max pooling + conv for channel reduction
        self.downsamples = nn.ModuleList([nn.Sequential(nn.MaxPool2d(2 ** (len(e_chans) - idx)),
                                                        nn.Conv2d(e_chan, base_chan, 3, padding=1))
                                          for idx, e_chan in enumerate(e_chans)])
        self.skip = nn.Conv2d(skip_chan, base_chan, 3, padding=1) if len(e_chans) > 0 else nn.Identity()
        # Upsample = bilinear interpolation + conv for channel reduction
        self.upsamples = nn.ModuleList([nn.Sequential(nn.Upsample(scale_factor=2 ** (idx + 1),
                                                                  mode='bilinear', align_corners=True),
                                                      nn.Conv2d(d_chan, base_chan, 3, padding=1))
                                        for idx, d_chan in enumerate(d_chans)])

        self.block = nn.Sequential(*conv_sequence(depth * base_chan, depth * base_chan,
                                                  act_layer, norm_layer, drop_layer, conv_layer,
                                                  kernel_size=3, padding=1))

    def forward(self, downfeats: List[Tensor], feat: Tensor, upfeats: List[Tensor]):

        if len(downfeats) != len(self.downsamples) or len(upfeats) != len(self.upsamples):
            raise ValueError(f"Expected {len(self.downsamples)} encoding & {len(self.upsamples)} decoding features, "
                             f"received: {len(downfeats)} & {len(upfeats)}")

        # Concatenate full-scale features
        x = torch.cat((*[downsample(downfeat) for downsample, downfeat in zip(self.downsamples, downfeats)],
                       self.skip(feat),
                       *[upsample(upfeat) for upsample, upfeat in zip(self.upsamples, upfeats)]), dim=1)

        return self.block(x)


class UNet3p(nn.Module):
    """Implements a UNet3+ architecture

    Args:
        layout (list<int>): number of channels after each contracting block
        in_channels (int, optional): number of channels in the input tensor
        num_classes (int, optional): number of output classes
        act_layer: activation layer
        norm_layer: normalization layer
        drop_layer: dropout layer
        conv_layer: convolutional layer
    """
    def __init__(
        self,
        layout: List[int],
        in_channels: int = 3,
        num_classes: int = 10,
        act_layer: Optional[nn.Module] = None,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        drop_layer: Optional[Callable[..., nn.Module]] = None,
        conv_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super().__init__()

        if act_layer is None:
            act_layer = nn.ReLU(inplace=True)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        # Contracting path
        self.encoders = nn.ModuleList([])
        _layout = [in_channels] + layout
        _pool = False
        for in_chan, out_chan in zip(_layout[:-1], _layout[1:]):
            self.encoders.append(down_path(in_chan, out_chan, _pool, 1, act_layer, norm_layer, drop_layer, conv_layer))
            _pool = True

        # Expansive path
        self.decoders = nn.ModuleList([])
        for row in range(len(layout) - 1):
            self.decoders.append(FSAggreg(layout[:row],
                                          layout[row],
                                          [len(layout) * layout[0]] * (len(layout) - 2 - row) + layout[-1:],
                                          act_layer, norm_layer, drop_layer, conv_layer))

        # Classifier
        self.classifier = nn.Conv2d(len(layout) * layout[0], num_classes, 1)

        init_module(self, 'relu')

    def forward(self, x: Tensor) -> Tensor:

        xs: List[Tensor] = []
        # Contracting path
        for encoder in self.encoders:
            xs.append(encoder(xs[-1] if len(xs) > 0 else x))

        # Full-scale expansive path
        for idx in range(len(self.decoders) - 1, -1, -1):
            xs[idx] = self.decoders[idx](xs[:idx], xs[idx], xs[idx + 1:])

        # Classifier
        x = self.classifier(xs[0])
        return x


def _unet(arch: str, pretrained: bool, progress: bool, **kwargs: Any) -> nn.Module:
    # Retrieve the correct Darknet layout type
    unet_type = sys.modules[__name__].__dict__[default_cfgs[arch]['arch']]
    # Build the model
    model = unet_type(default_cfgs[arch]['layout'], **kwargs)
    # Load pretrained parameters
    if pretrained:
        load_pretrained_params(model, default_cfgs[arch]['url'], progress)

    return model


def unet(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> UNet:
    """U-Net from
    `"U-Net: Convolutional Networks for Biomedical Image Segmentation" <https://arxiv.org/pdf/1505.04597.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        torch.nn.Module: semantic segmentation model
    """

    return _unet('unet', pretrained, progress, **kwargs)  # type: ignore[return-value]


def unetp(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> UNetp:
    """UNet+ from `"UNet++: Redesigning Skip Connections to Exploit Multiscale Features in Image Segmentation"
    <https://arxiv.org/pdf/1912.05074.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        torch.nn.Module: semantic segmentation model
    """

    return _unet('unetp', pretrained, progress, **kwargs)  # type: ignore[return-value]


def unetpp(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> UNetpp:
    """UNet++ from `"UNet++: Redesigning Skip Connections to Exploit Multiscale Features in Image Segmentation"
    <https://arxiv.org/pdf/1912.05074.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        torch.nn.Module: semantic segmentation model
    """

    return _unet('unetpp', pretrained, progress, **kwargs)  # type: ignore[return-value]


def unet3p(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> UNet3p:
    """UNet3+ from
    `"UNet 3+: A Full-Scale Connected UNet For Medical Image Segmentation" <https://arxiv.org/pdf/2004.08790.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        torch.nn.Module: semantic segmentation model
    """

    return _unet('unet3p', pretrained, progress, **kwargs)  # type: ignore[return-value]
