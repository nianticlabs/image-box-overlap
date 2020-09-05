"""
Predicting Visual Overlap of Images Through Interpretable Non-Metric Box Embeddings.

Datasets util functions.
We thank Carl Toft for helping with normal computation code during his internship at Niantic.

Copyright Niantic 2020. Patent Pending. All rights reserved.

This software is licensed under the terms of the Image-box-overlap licence
which allows for non-commercial use only, the full terms of which are made
available in the LICENSE file.
"""

import os
import numpy as np
import torch
import PIL
import h5py
import pickle
from scipy.spatial.transform import Rotation


def backproject_depthmap(depths, inv_k):
    """ Projects depth maps into 3D space
    :param depths: Input depths with shape [batch_size x height x width]
    :param inv_k: Inverse intrinsics with shape [batch_size x 3 x 3]
    :return: 3D points for each pixel in a depth map with shape [batch_size x height x width x 3]
    """

    batch_size = depths.shape[0]

    yy, xx = torch.meshgrid([torch.arange(depths.shape[1], dtype=depths.dtype, device=depths.device),
                             torch.arange(depths.shape[2], dtype=depths.dtype, device=depths.device)])

    coords = torch.cat([
        xx.reshape(-1, 1),
        yy.reshape(-1, 1),
        torch.ones([depths.shape[1] * depths.shape[2], 1], dtype=depths.dtype, device=depths.device)], 1)

    coords = coords.reshape(1, -1, 3, 1).repeat(batch_size, 1, 1, 1)
    inv_k = inv_k.reshape(-1, 1, 3, 3)
    point3d = torch.matmul(inv_k, coords)
    point3d = point3d * depths.reshape(batch_size, -1, 1, 1)
    point3d = point3d.reshape(batch_size, depths.shape[1], depths.shape[2], 3)
    return point3d


def get_normals(points3d):
    """ Computes normal maps from 3D cloud
    :param points3d: Ordered 3D cloud with shape [batch_size x height x width x 3]
    :return: Normal map with shape [batch_size x height x width x 3]
    """
    batch_size = points3d.shape[0]
    dtype = points3d.dtype
    device = points3d.device
    pi = torch.tensor([np.pi], dtype=dtype, device=device)
    nan = torch.tensor([np.nan], dtype=dtype, device=device)

    # Create a huge tensor to keep the points, as well as the points shifted one step in all directions
    hugeTensor = torch.zeros(batch_size,
                             points3d.shape[1] + 2, points3d.shape[2] + 2, points3d.shape[3], 9,
                             dtype=points3d.dtype, device=device)
    hugeTensor[:, 1:points3d.shape[1] + 1, 1:points3d.shape[2] + 1, 0:3, 0] = points3d  # original points in center

    # Now add them in this order: top-left, top-center, top-right
    hugeTensor[:, 0:points3d.shape[1], 0:points3d.shape[2], 0:3, 1] = points3d
    hugeTensor[:, 0:points3d.shape[1], 1:points3d.shape[2] + 1, 0:3, 2] = points3d
    hugeTensor[:, 0:points3d.shape[1], 2:points3d.shape[2] + 2, 0:3, 3] = points3d

    # Now center-left, center-right
    hugeTensor[:, 1:points3d.shape[1] + 1, 0:points3d.shape[2], 0:3, 4] = points3d
    hugeTensor[:, 1:points3d.shape[1] + 1, 2:points3d.shape[2] + 2, 0:3, 5] = points3d

    # Now bottom-left, bottom-cente1, bottom-right
    hugeTensor[:, 2:points3d.shape[1] + 2, 0:points3d.shape[2], 0:3, 6] = points3d
    hugeTensor[:, 2:points3d.shape[1] + 2, 1:points3d.shape[2] + 1, 0:3, 7] = points3d
    hugeTensor[:, 2:points3d.shape[1] + 2, 2:points3d.shape[2] + 2, 0:3, 8] = points3d

    # Done! Now compute the mean vector for each pixel.
    meanPoints = hugeTensor.mean(4)
    h = hugeTensor.shape[1]
    w = hugeTensor.shape[2]

    S = torch.zeros(batch_size, h, w, 3, 3, dtype=dtype, device=device)
    for k in range(9):
        hugeTensor[:, :, :, :, k] = hugeTensor[:, :, :, :, k] - meanPoints
        S = S + hugeTensor[:, :, :, :, k].reshape(batch_size, h, w, 1, 3) * hugeTensor[:, :, :, :, k].reshape(
            batch_size, h, w, 3, 1)
    S = S / 8

    eye3 = torch.eye(3, dtype=dtype, device=device).unsqueeze(0)
    # Now, compute the smallest eigenvector of each covariance matrix
    mat = torch.tensor(np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]]), dtype=dtype, device=device)
    mat = mat.unsqueeze(0)
    p1 = ((S * mat) ** 2).sum(4).sum(3)
    q = (S * eye3).sum(4).sum(3) / 3
    p2 = ((((S - q.reshape(batch_size, q.shape[1], q.shape[2], 1, 1)) * eye3) * eye3) ** 2).sum(4).sum(3)
    p2 = p2 + 2 * p1
    val_mask = p2 > 0
    p2[~val_mask] = 1

    p2 = torch.sqrt(p2 / 6.0)

    # No more division by zero
    B = (S - q.reshape(batch_size, q.shape[1], q.shape[2], 1, 1) * eye3) / p2.reshape(batch_size, p2.shape[1],
                                                                                      p2.shape[2], 1, 1)

    detB = B[:, :, :, 0, 0] * B[:, :, :, 1, 1] * B[:, :, :, 2, 2] + \
           B[:, :, :, 0, 1] * B[:, :, :, 1, 2] * B[:, :, :, 2, 0] + \
           B[:, :, :, 0, 2] * B[:, :, :, 1, 0] * B[:, :, :, 2, 1] - \
           B[:, :, :, 0, 2] * B[:, :, :, 1, 1] * B[:, :, :, 2, 0] - \
           B[:, :, :, 0, 1] * B[:, :, :, 1, 0] * B[:, :, :, 2, 2] - \
           B[:, :, :, 0, 0] * B[:, :, :, 1, 2] * B[:, :, :, 2, 1]

    detB = detB / 2.0
    detB = detB.clamp(min=-1, max=1)
    lambda0 = 2.0 * p2 * torch.cos(torch.acos(detB) / 3.0 + 2 * pi / 3.0) + q

    detM = (S[:, :, :, 0, 0] - lambda0) * (S[:, :, :, 1, 1] - lambda0) - S[:, :, :, 0, 1] * S[:, :, :, 1, 0]
    normals = torch.ones(batch_size, lambda0.shape[1], lambda0.shape[2], 3, dtype=dtype, device=device)

    val_mask = val_mask & (detM != 0)
    detM[~val_mask] = 1
    normals[:, :, :, 0] = (S[:, :, :, 0, 2] * (lambda0 - S[:, :, :, 1, 1]) + S[:, :, :, 1, 2] * S[:, :, :, 0, 1]) / detM
    normals[:, :, :, 1] = (S[:, :, :, 1, 2] * (lambda0 - S[:, :, :, 0, 0]) + S[:, :, :, 0, 2] * S[:, :, :, 1, 0]) / detM

    normals_length = torch.sqrt((normals ** 2).sum(3))
    normals = normals / normals_length.reshape(batch_size, normals.shape[1], normals.shape[2], 1)

    neg_offsets = ((normals * meanPoints).sum(3)).reshape(batch_size, normals.shape[1], normals.shape[2], 1)
    normals = normals * torch.sign(neg_offsets)

    normals[~val_mask] = nan
    lambda0[~val_mask] = nan

    normals = normals[:, 1:normals.shape[1] - 1, 1:normals.shape[2] - 1, :]
    return normals, lambda0


def compute_nso(subset_1, subset_2, threshold, normals_1, normals_2):
    """ Computes normalized surface overlap from two input point clouds and their normals.
    :param subset_1: 3D cloud from image 1 (or 2) with shape [3 x num_points]
    :param threshold: Maximum distance for two points to be considered as overlapping.
    :param normals_1: Normals for image 1 (or 2) with shape [3 x num_points]
    :return: enclosure and concentration
    """

    # Compute the distance between each pair of pixels and find smallest distance
    set_1 = subset_1.reshape((3, -1, 1))
    set_2 = subset_2.reshape((3, 1, -1))

    diff = set_1 - set_2
    dist = np.sqrt(np.sum(np.square(diff), 0))

    s1_nns = np.argmin(dist, 1)
    s2_nns = np.argmin(dist, 0)

    s1_nn_dist = np.min(dist, 1)
    s2_nn_dist = np.min(dist, 0)

    # Compute cosine between normals of each point from cloud 1 and its nearest neighbor from cloud 2
    # and scale to interval [0,1]. Then vice-versa.
    v1 = normals_1
    v2 = normals_2[:, s1_nns]

    weights_1 = 0.5 * (np.sum(np.multiply(v1, v2), 0) + 1.0)
    weights_1[np.isnan(weights_1)] = 0.

    v1 = normals_2
    v2 = normals_1[:, s2_nns]

    weights_2 = 0.5 * (np.sum(np.multiply(v1, v2), 0) + 1.0)
    weights_2[np.isnan(weights_2)] = 0.

    # Compute weighted ratio of pixels with neighbor in distance smaller than threshold
    weighted_s1_has_nn = np.matmul((s1_nn_dist < threshold), weights_1)
    weighted_s2_has_nn = np.matmul((s2_nn_dist < threshold), weights_2)

    nso_x2y = weighted_s1_has_nn / set_1.shape[1]
    nso_y2x = weighted_s2_has_nn / set_2.shape[2]

    nso_dict = {
        'x->y': nso_x2y,
        'y->x': nso_y2x,
    }

    return nso_dict


def process_surface_overlap_batch(batch, normals_folder, threshold, num_points):

    # Load precomputed normal maps and 3D clouds (output from compute_normals.py)
    normals_path1 = os.path.join(normals_folder, batch['image_id'][0] + '.pickle')
    normals_path2 = os.path.join(normals_folder, batch['image_id'][1] + '.pickle')

    with open(normals_path1, 'rb') as f:
        pick1 = pickle.load(f)
    with open(normals_path2, 'rb') as f:
        pick2 = pickle.load(f)

    normals_1 = pick1['normals'].transpose(2, 0, 1).reshape((3, -1))
    normals_2 = pick2['normals'].transpose(2, 0, 1).reshape((3, -1))

    coords_1 = pick1['points3d'].transpose(2, 0, 1).reshape((3, -1))
    coords_2 = pick2['points3d'].transpose(2, 0, 1).reshape((3, -1))

    coords_1 = np.concatenate([coords_1, np.ones((1, coords_1.shape[1]))], 0)
    coords_2 = np.concatenate([coords_2, np.ones((1, coords_2.shape[1]))], 0)

    # Delete invalid pixels with zero depth
    valid_coords_1 = coords_1[:, coords_1[2, :] > 0]
    valid_coords_2 = coords_2[:, coords_2[2, :] > 0]

    valid_normals_1 = normals_1[:, coords_1[2, :] > 0]
    valid_normals_2 = normals_2[:, coords_2[2, :] > 0]

    world_points_1 = cam2world(valid_coords_1, np.linalg.inv(batch['camera_pose'][0]))
    world_points_2 = cam2world(valid_coords_2, np.linalg.inv(batch['camera_pose'][1]))

    # Sample points for efficiency (if needed)
    if world_points_1.shape[1] > num_points:
        indices1 = np.random.choice(world_points_1.shape[1], size=num_points, replace=False)
        indices2 = np.random.choice(world_points_2.shape[1], size=num_points, replace=False)
    else:
        indices1 = np.random.choice(world_points_1.shape[1], size=world_points_1.shape[1], replace=False)
        indices2 = np.random.choice(world_points_2.shape[1], size=world_points_2.shape[1], replace=False)

    subset_1 = world_points_1[:3, indices1]
    subset_2 = world_points_2[:3, indices2]

    normals_1 = valid_normals_1[:, indices1]
    normals_2 = valid_normals_2[:, indices2]

    nso_dict = compute_nso(subset_1, subset_2, threshold, normals_1, normals_2)

    return nso_dict


def pil_loader(path):
    """
    Load image from path with PIL. PIL is used to avoid
    ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    """
    with open(path, 'rb') as file_handler:
        with PIL.Image.open(file_handler) as img:
            return img.convert('RGB')


def cam2world(coords, pose):
    return np.matmul(pose, coords)


def params2matrix(camera_params, resized_size, orig_size):
    """ From MegaDepth annotated camera parameter format to intrisincs matrix.
    Specific for MegaDepth: it also adjusts parameters taking into account discrepancy in image size between
    original color image (annotated) and aligned to depth images. More details in supplementary material.
    """
    scale_x = resized_size[0] / orig_size[0]
    scale_y = resized_size[1] / orig_size[1]

    return np.array([[scale_x * camera_params[0], scale_x * camera_params[3], scale_x * camera_params[1]],
                     [0., scale_y * camera_params[0], scale_y * camera_params[2]],
                     [0., 0., 1.]])


def quat2mat(q, t):
    """
    Quaternion to rotation and translation in matrix form
    """
    p_mat = np.zeros((4, 4))
    r_mat = Rotation.from_quat(q).as_matrix()
    p_mat[:3, :3] = r_mat
    p_mat[:3, -1] = t
    p_mat[-1, -1] = 1.0
    return p_mat


def read_depth_megadepth(filepath):
    hdf5_file_read = h5py.File(filepath, 'r')
    depth = hdf5_file_read.get('/depth')
    depth = np.array(depth)
    hdf5_file_read.close()
    return depth
