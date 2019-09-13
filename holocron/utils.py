#!usr/bin/python
# -*- coding: utf-8 -*-

"""
Utils
"""

import numpy as np
import torch
from PIL import Image
from matplotlib import cm


class ActivationMapper(object):
    """Implements a class activation map extractor as described in https://arxiv.org/abs/1512.04150

    Args:
        model (torch.nn.Module): input model
        conv_layer (str): name of the last convolutional layer
        fc_layer (str): name of the fully connected layer
    """

    conv_fmap = None

    def __init__(self, model, conv_layer, fc_layer):

        if not hasattr(model, conv_layer) or not hasattr(model, fc_layer):
            raise ValueError(f"Unable to find submodules {conv_layer} and {fc_layer} in the model")
        self.conv_layer = conv_layer
        self.fc_layer = fc_layer
        self.model = model
        # Forward hook
        self.model._modules.get(self.conv_layer).register_forward_hook(self.__hook)
        # Softmax weight
        self.smax_weights = self.model._modules.get(fc_layer).weight.data

    def __hook(self, module, input, output):
        self.conv_fmap = output.data

    def get_activation_maps(self, class_idxs, normalized=True):
        """Recreate class activation maps

        Args:
            class_idxs (list<int>): class indices for expected activation maps
            normalized (bool): should the activation map be normalized

        Returns:
            batch_cams (torch.Tensor<float>): activation maps of the last forwarded batch
        """

        if any(idx >= self.smax_weights.size(0) for idx in class_idxs):
            raise ValueError("Expected class_idx to be lower than number of output classes")

        if self.conv_fmap is None:
            raise TypeError("Inputs need to be forwarded in the model for the conv features to be hooked")

        # Flatten spatial dimensions of feature map
        batch_cams = self.smax_weights[class_idxs, :] @ torch.flatten(self.conv_fmap, 2)
        # Normalize feature map
        if normalized:
            batch_cams -= batch_cams.min(dim=2, keepdim=True)[0]
            batch_cams /= batch_cams.max(dim=2, keepdim=True)[0]

        return batch_cams.view(self.conv_fmap.size(0), len(class_idxs), self.conv_fmap.size(3), self.conv_fmap.size(2)).cpu()


def overlay_mask(img, mask, colormap='jet', alpha=0.7):
    """Overlay a colormapped mask on a background image

    Args:
        img (PIL.Image.Image): background image
        mask (PIL.Image.Image): mask to be overlayed in grayscale
        colormap (str): colormap to be applied on the mask
        alpha (float): transparency of the background image

    Returns:
        overlayed_img (PIL.Image.Image): overlayed image
    """

    if not isinstance(img, Image.Image) or not isinstance(mask, Image.Image):
        raise TypeError('img and mask arguments need to be PIL.Image')

    if not isinstance(alpha, float) or alpha < 0 or alpha >= 1:
        raise ValueError('alpha argument is expected to be of type float between 0 and 1')

    cmap = cm.get_cmap(colormap)
    # Resize mask and apply colormap
    overlay = mask.resize(img.size, resample=Image.BICUBIC)
    overlay = (255 * cmap(np.asarray(overlay) ** 2)[:, :, 1:]).astype(np.uint8)
    # Overlay the image with the mask
    overlayed_img = Image.fromarray((alpha * np.asarray(img) + (1 - alpha) * overlay).astype(np.uint8))

    return overlayed_img
