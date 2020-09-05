"""
Predicting Visual Overlap of Images Through Interpretable Non-Metric Box Embeddings

Datasets options parser.

Copyright Niantic 2020. Patent Pending. All rights reserved.

This software is licensed under the terms of the Image-box-overlap licence
which allows for non-commercial use only, the full terms of which are made
available in the LICENSE file.
"""

import argparse


class NormalsComputeOptions:
    def __init__(self):
        self.options = None
        self.parser = argparse.ArgumentParser(description='Compute Normals MegaDepth')
        self.parser.add_argument('--dataset_json', default='data/dataset_jsons/megadepth/bigben.json',
                                 help="Path to dataset json files.")
        self.parser.add_argument('--output_folder',
                                 help="Path to save normals.")
        self.parser.add_argument('--batch_size', type=int, default=1,
                                 help="Number of images to compute normals in parallel."
                                      "Important! For MegaDepth it only supports one image per batch.")
        self.parser.add_argument('--num_workers', type=int, default=8,
                                 help="Number of workers for data loading (default: 8).")

    def parse(self, *args, **kwargs):
        self.options = self.parser.parse_args(*args, **kwargs)
        return self.options


class OverlapComputeOptions:
    def __init__(self):
        self.options = None
        self.parser = argparse.ArgumentParser(description='Compute Overlap MegaDepth (Requires Normals)')
        self.parser.add_argument('--normals_folder', help="Path to store normal data.")
        self.parser.add_argument('--dataset_json', default='data/dataset_jsons/megadepth/bigben.json',
                                 help="Path to dataset json files.")
        self.parser.add_argument('--num_pairs', type=int, default=10000,
                                 help="Number of image pairs to compute surface (default: 10000).")
        self.parser.add_argument('--num_sampled_points', type=int, default=5000,
                                 help='Number of points to be sampled for overlap computation (default: 5000).')
        self.parser.add_argument('--threshold', type=float, default=0.1,
                                 help='Maximum distance for two points to be considered as overlapping (default: 0.1).')
        self.parser.add_argument('--num_workers', type=int, default=8,
                                 help="Number of workers for data loading (default: 8).")

    def parse(self, *args, **kwargs):
        self.options = self.parser.parse_args(*args, **kwargs)
        return self.options
