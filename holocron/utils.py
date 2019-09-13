#!usr/bin/python
# -*- coding: utf-8 -*-

"""
Utils
"""

import torch


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
