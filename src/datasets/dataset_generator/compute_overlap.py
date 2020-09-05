"""
Predicting Visual Overlap of Images Through Interpretable Non-Metric Box Embeddings.

Normalized surface overlap dataset generator.
It requires precomputed normals with compute_normals.py.

Copyright Niantic 2020. Patent Pending. All rights reserved.

This software is licensed under the terms of the Image-box-overlap licence
which allows for non-commercial use only, the full terms of which are made
available in the LICENSE file.
"""

import os
import torch
from tqdm import tqdm
from .options import OverlapComputeOptions
from .utils import process_surface_overlap_batch
from ..megadepth_loader import MegaDepthImageLoader


def main():
    opts = OverlapComputeOptions().parse()

    dataset = MegaDepthImageLoader(opts.dataset_json, mode='data_generation')
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True,
                                              num_workers=opts.num_workers, pin_memory=True)

    output_file = os.path.join(opts.normals_folder, 'computed_overlaps.txt')
    writer = open(output_file, 'w')

    print(f"Computing the surface overlap of {opts.num_pairs} image pairs from previously computed normals...")
    for _ in tqdm(range(opts.num_pairs)):
        batch = next(iter(data_loader))
        nso_batch = process_surface_overlap_batch(batch,
                                                  opts.normals_folder,
                                                  opts.threshold,
                                                  opts.num_sampled_points)
        if nso_batch:
            writer.write(f"{batch['image_id'][0]} {batch['image_id'][1]} {str(nso_batch['x->y'])}\n")
            writer.write(f"{batch['image_id'][1]} {batch['image_id'][0]} {str(nso_batch['y->x'])}\n")

    writer.close()


if __name__ == '__main__':
    main()