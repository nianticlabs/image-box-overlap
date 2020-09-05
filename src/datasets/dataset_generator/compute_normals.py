"""
Predicting Visual Overlap of Images Through Interpretable Non-Metric Box Embeddings.

Compute and store surface normals.

Copyright Niantic 2020. Patent Pending. All rights reserved.

This software is licensed under the terms of the Image-box-overlap licence
which allows for non-commercial use only, the full terms of which are made
available in the LICENSE file.
"""

import os
import numpy as np
import torch
import pickle
from tqdm import tqdm
from .options import NormalsComputeOptions
from .utils import backproject_depthmap, get_normals, read_depth_megadepth
from ..megadepth_loader import MegaDepthImageLoader


def process_normals_batch(batch, batch_size=1):
    depth_images, k_mat, inv_k_mat, file_names_depth = [], [], [], []
    for i in range(batch_size):
        inv_k_mat.append(torch.tensor(np.linalg.inv(batch['camera_calib'][i])))
        depth_images.append(torch.tensor(read_depth_megadepth(batch['depth_path'][i])))

    depth_images = torch.stack(depth_images)
    inv_k_mat = torch.stack(inv_k_mat).float()

    if torch.cuda.is_available():
        inv_k_mat = inv_k_mat.cuda()
        depth_images = depth_images.cuda()

    with torch.no_grad():
        points3d = backproject_depthmap(depth_images, inv_k_mat)
        normals, plane_fit_error = get_normals(points3d)

    normals = normals.cpu().detach().numpy().astype(np.float32)
    points3d = points3d.cpu().detach().numpy().astype(np.float32)
    plane_fit_error = plane_fit_error.cpu().detach().numpy().astype(np.float32)

    return normals, points3d, plane_fit_error


def main():
    opts = NormalsComputeOptions().parse()
    dataset = MegaDepthImageLoader(opts.dataset_json, mode='data_generation')
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=opts.batch_size,
                                              shuffle=False,
                                              num_workers=opts.num_workers,
                                              pin_memory=True)

    if not os.path.exists(opts.output_folder):
        os.makedirs(opts.output_folder)

    print(f"Computing and saving normals of a total of {len(data_loader)} images...")
    for batch_idx, batch in enumerate(tqdm(data_loader)):
        normals, points3d, plane_fit_error = process_normals_batch(batch,
                                                                   opts.batch_size)
        for i in range(opts.batch_size):
            result_dict_i = {
                'normals': normals[i],
                'points3d': points3d[i],
                'plane_fit_error': plane_fit_error[i]
            }
            result_path_i = os.path.join(opts.output_folder,
                                         f"{batch['image_id'][i]}.pickle")

            with open(result_path_i, 'wb') as f:
                pickle.dump(result_dict_i, f)


if __name__ == '__main__':
    main()
