
import torch
import torch.nn as nn

__all__ = ['SAM']


class SAM(nn.Module):
    """SAM layer from `"CBAM: Convolutional Block Attention Module" <https://arxiv.org/pdf/1807.06521.pdf>`_
    modified in `"YOLOv4: Optimal Speed and Accuracy of Object Detection" <https://arxiv.org/pdf/2004.10934.pdf>`_.

    Args:
        in_channels (int): input channels
    """
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(self.conv(x))
