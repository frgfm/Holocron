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
    as well as the GradCAM extractor as described in https://arxiv.org/pdf/1610.02391.pdf

    Args:
        model (torch.nn.Module): input model
        conv_layer (str): name of the last convolutional layer
    """

    hook_a, hook_g = None, None

    def __init__(self, model, conv_layer):

        if not hasattr(model, conv_layer):
            raise ValueError(f"Unable to find submodule {conv_layer} in the model")
        self.model = model
        # Forward hook
        self.model._modules.get(conv_layer).register_forward_hook(self.__hook_a)
        # Backward hook
        self.model._modules.get(conv_layer).register_backward_hook(self.__hook_g)

    def __hook_a(self, module, input, output):
        self.hook_a = output.data

    def __hook_g(self, module, input, output):
        self.hook_g = output[0].data

    def get_activation_maps(self, output, class_idx, normalized=True):
        """Recreate class activation maps

        Args:
            output (torch.Tensor[N, K]): output of the hooked model
            class_idx (int): class index for expected activation map
            normalized (bool, optional): should the activation map be normalized

        Returns:
            torch.Tensor[N, H, W]: activation maps of the last forwarded batch at the hooked layer
        """

        if self.hook_a is None:
            raise TypeError("Inputs need to be forwarded in the model for the conv features to be hooked")

        # One-hot encode the expected class
        one_hot = torch.zeros(output.shape[-1], dtype=torch.float32)
        one_hot[class_idx] = 1
        one_hot.requires_grad_(True).to(output.device)

        # Backpropagate to get the gradients on the hooked layer
        loss = (one_hot.unsqueeze(0) * output).sum()
        self.model.zero_grad()
        loss.backward(retain_graph=True)

        # Global average pool the gradients over spatial dimensions
        weights = self.hook_g.data.mean(axis=(2, 3))
        # Get the feature activation map
        fmap = self.hook_a.data
        # Perform the weighted combination to get the CAM
        batch_cams = torch.relu((weights.view(*weights.shape, 1, 1) * fmap).sum(dim=1))

        # Normalize the CAM
        if normalized:
            batch_cams -= batch_cams.flatten(start_dim=1).min().view(-1, 1, 1)
            batch_cams /= batch_cams.flatten(start_dim=1).max().view(-1, 1, 1)

        return batch_cams


def overlay_mask(img, mask, colormap='jet', alpha=0.7):
    """Overlay a colormapped mask on a background image

    Args:
        img (PIL.Image.Image): background image
        mask (PIL.Image.Image): mask to be overlayed in grayscale
        colormap (str, optional): colormap to be applied on the mask
        alpha (float, optional): transparency of the background image

    Returns:
        PIL.Image.Image: overlayed image
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


def get_module_names(module, prefix=''):
    """Recursively gets all the module's children names

    Args:
        module (torch.nn.Module): input module
        prefix (str, optional): name of the current module
    Returns:
        list<str>: list of module names
    """
    # Add a full stop between parent and children names
    if len(prefix) > 0:
        prefix += '.'
    names = []
    for n, c in module.named_children():
        current = f"{prefix}{n}"
        # Get submodules names
        if any(c.children()):
            names.extend(get_module_names(c, prefix=current))
        # Add leaf name
        else:
            names.append(current)
    return names
