"""
Predicting Visual Overlap of Images Through Interpretable Non-Metric Box Embeddings.

Util functions. It includes important functions as box overlap computation.

Copyright Niantic 2020. Patent Pending. All rights reserved.

This software is licensed under the terms of the Image-box-overlap licence
which allows for non-commercial use only, the full terms of which are made
available in the LICENSE file.
"""

from __future__ import absolute_import, division, print_function
import os
import numpy as np
import torch
import hashlib
import zipfile
from six.moves import urllib
from torch.nn import functional


def box_overlap_soft(bx_center, bx_extent, by_center, by_extent, box_rho=5):
    """
    Computes soft box overlap of two boxes in D-dimensional space
    :param bx_center: D-dimensional vector with box x center coordinates
    :param bx_extent: D-dimensional vector with box x size
    :param by_center: D-dimensional vector with box y center coordinates
    :param by_extent: D-dimensional vector with box y size
    :param box_rho: temperature parameter, $\rho$ in the paper
    :return: box overlap of bx and by
    """
    bx_min, by_min = bx_center - 0.5 * bx_extent, by_center - 0.5 * by_extent
    bx_max, by_max = bx_min + bx_extent, by_min + by_extent

    lower_upper_bound = torch.min(torch.stack([bx_max, by_max], dim=-1), dim=-1)[0]
    upper_lower_bound = torch.max(torch.stack([bx_min, by_min], dim=-1), dim=-1)[0]

    flat_overlap = lower_upper_bound - upper_lower_bound

    area_x_exp = torch.exp(
        torch.sum(torch.log(functional.softplus((bx_max - bx_min) * box_rho) / box_rho + 1e-10), 1))
    intersect_exp = torch.exp(
        torch.sum(torch.log(functional.softplus(flat_overlap * box_rho) / box_rho + 1e-10), 1))

    return intersect_exp / area_x_exp


def box_overlap(bx_center, bx_extent, by_center, by_extent):
    """
    Computes box overlap of two boxes in D-dimensional space
    :param bx_center: D-dimensional vector with box x center coordinates
    :param bx_extent: D-dimensional vector with box x size
    :param by_center: D-dimensional vector with box y center coordinates
    :param by_extent: D-dimensional vector with box y size
    :return: box overlap of bx and by
    """
    bx_min, by_min = bx_center - 0.5 * bx_extent, by_center - 0.5 * by_extent
    bx_max, by_max = bx_min + bx_extent, by_min + by_extent

    lower_upper_bound = torch.min(torch.stack([bx_max, by_max], dim=-1), dim=-1)[0]
    upper_lower_bound = torch.max(torch.stack([bx_min, by_min], dim=-1), dim=-1)[0]

    intersection = torch.prod(lower_upper_bound - upper_lower_bound, dim=-1)
    area_x = torch.prod(bx_max - bx_min, dim=-1)

    zeros_tensor = torch.zeros(intersection.shape).type_as(intersection)

    return torch.max(torch.stack([zeros_tensor, intersection / (area_x + 1e-10)]), dim=0)[0]


def compute_relative_scale(enclosure, concentration, im_x_size, im_y_size):
    """
    Computes relative scale between two images using predicted enclosure and concentration
    """
    x_width, x_height = im_x_size
    y_width, y_height = im_y_size

    x_area = x_width * x_height
    y_area = y_width * y_height

    x_scale, y_scale = 1., 1.

    ratio = enclosure / concentration

    y_scale = np.sqrt(ratio * x_area / y_area)

    if y_scale > 1.:  # we want to downsample, not upsample
        x_scale = 1. / y_scale
        y_scale = 1.

    return x_scale, y_scale


def download_model_if_doesnt_exist(model_name):
    """
    If pretrained MegaDepth model doesn't exist, download and unzip it
    """
    # values are tuples of (<google cloud URL>, <md5 checksum>)
    download_paths = {
        'bigben':
            ('https://storage.googleapis.com/niantic-lon-static/research/image-box-overlap/models/megadepth/bigben.zip',
             'df43091df376c70e9701ee4db4f903ef'),
        'notredame':
            ('https://storage.googleapis.com/niantic-lon-static/research/image-box-overlap/models/megadepth/notredame'
             '.zip',
             '2c228d48cac1f911b53b44de190633b6'),
        'venice':
            ('https://storage.googleapis.com/niantic-lon-static/research/image-box-overlap/models/megadepth/venice.zip',
             '2996a6e846cb94333ccd83a77445b913'),
        'florence':
            ('https://storage.googleapis.com/niantic-lon-static/research/image-box-overlap/models/megadepth/florence'
             '.zip',
             '084194eb90a7b3e49b542e7b8e11451e'),
        }

    if not os.path.exists('models'):
        os.makedirs('models')

    model_path = os.path.join('models', model_name)

    def check_file_matches_md5(checksum, fpath):
        if not os.path.exists(fpath):
            return False
        with open(fpath, 'rb') as f:
            current_md5checksum = hashlib.md5(f.read()).hexdigest()
        return current_md5checksum == checksum

    # See if we have the model already downloaded...
    if not os.path.exists(os.path.join(model_path, f'{model_name}.pth.tar')):

        model_url, required_md5checksum = download_paths[model_name]

        if not check_file_matches_md5(required_md5checksum, f'{model_path}.zip'):
            print(f"-> Downloading pretrained model to {model_path}.zip")
            urllib.request.urlretrieve(model_url, f'{model_path}.zip')

        if not check_file_matches_md5(required_md5checksum, f'{model_path}.zip'):
            print("   Failed to download a file which matches the checksum - quitting")
            quit()

        print("   Unzipping model...")
        with zipfile.ZipFile(f'{model_path}.zip', 'r') as f:
            f.extractall(model_path)

        print(f"   Model unzipped to {model_path}")


def checkpoint_loader(ckpt_path):
    """
    Returns saved weights providing compatibility with saved models to reproduce paper results.
    """
    checkpoint = torch.load(ckpt_path)
    # trained models for ECCV paper using PyTorch (.pth.tar extension)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    # trained models using new implementation with Lightning (.cpkt extension)
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        state_dict_new = {}
        for key, value in state_dict.items():
            # lightning adds the prefix "model" to the module name layers, removing that
            new_key = key[6:]
            state_dict_new[new_key] = value
            state_dict = state_dict_new
    else:
        raise ModelVersionException
    return state_dict


class ModelVersionException(Exception):
    def __init__(self, msg="Incompatible model version. Use provided ones or the ones trained with the provided code",
                 *args, **kwargs):
        super().__init__(msg, *args, **kwargs)
