"""
Predicting Visual Overlap of Images Through Interpretable Non-Metric Box Embeddings.

Network model.

Copyright Niantic 2020. Patent Pending. All rights reserved.

This software is licensed under the terms of the Image-box-overlap licence
which allows for non-commercial use only, the full terms of which are made
available in the LICENSE file.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ResnetEncoder(nn.Module):
    def __init__(self, num_layers, box_ndim):
        super(ResnetEncoder, self).__init__()
        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        self.last_layer = 512
        if num_layers == 50:
            self.last_layer = 2048
        elif num_layers == 101:
            self.last_layer = 2048

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        encoder = resnets[num_layers](True)
        self.encoder = encoder

        self.layer0 = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.layer1 = nn.Sequential(encoder.maxpool, encoder.layer1)
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4
        self.fc1 = nn.Linear(self.last_layer // 4, box_ndim)
        self.fc2 = nn.Linear(self.last_layer // 4, box_ndim)
        self.fc = nn.Linear(self.last_layer, self.last_layer // 4)
        del encoder

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, input_image):
        # Normalize the input colorspace
        x = (input_image - 0.45) / 0.225

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.shape[0], x.shape[1], -1)
        x = torch.sum(x, 2) / x.shape[2]
        x = self.fc(x)
        x = F.relu(x)
        b1 = self.fc1(x)
        b2 = self.fc2(x)
        b2 = F.softplus(b2) + 0.1  # we want to avoid small boxes

        return b1, b2
