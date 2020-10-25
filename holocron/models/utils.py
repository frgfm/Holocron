import logging
import torch.nn as nn
from holocron.nn import BlurPool2d
from typing import List, Optional, Any, Callable, Tuple
from torchvision.models.utils import load_state_dict_from_url


__all__ = ['conv_sequence', 'load_pretrained_params']


def conv_sequence(
    in_channels: int,
    out_channels: int,
    act_layer: Optional[nn.Module] = None,
    norm_layer: Optional[Callable[[int], nn.Module]] = None,
    drop_layer: Optional[Callable[..., nn.Module]] = None,
    conv_layer: Optional[Callable[..., nn.Module]] = None,
    bn_channels: Optional[int] = None,
    attention_layer: Optional[Callable[[int], nn.Module]] = None,
    blurpool: bool = False,
    **kwargs: Any
) -> List[nn.Module]:

    if conv_layer is None:
        conv_layer = nn.Conv2d
    if bn_channels is None:
        bn_channels = out_channels

    conv_stride = kwargs.get('stride', 1)
    if blurpool and conv_stride > 1:
        kwargs['stride'] = 1

    conv_seq = [conv_layer(in_channels, out_channels, **kwargs)]

    if callable(norm_layer):
        conv_seq.append(norm_layer(bn_channels))
    if callable(act_layer):
        conv_seq.append(act_layer)
    if blurpool and conv_stride > 1:
        conv_seq.append(BlurPool2d(bn_channels, stride=conv_stride))
    if callable(attention_layer):
        conv_seq.append(attention_layer(bn_channels))
    if callable(drop_layer):
        conv_seq.append(drop_layer(inplace=True))

    return conv_seq


def load_pretrained_params(
    model: nn.Module,
    url: Optional[str] = None,
    progress: bool = True,
    key_replacement: Optional[Tuple[str, str]] = None,
    key_filter: Optional[str] = None,
) -> None:

    if url is None:
        logging.warning("Invalid model URL, using default initialization.")
    else:
        state_dict = load_state_dict_from_url(url, progress=progress)
        if isinstance(key_filter, str):
            state_dict = {k: v for k, v in state_dict.items() if k.startswith(key_filter)}
        if isinstance(key_replacement, tuple):
            state_dict = {k.replace(*key_replacement): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
