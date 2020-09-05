"""
Predicting Visual Overlap of Images Through Interpretable Non-Metric Box Embeddings

Options parser.

Copyright Niantic 2020. Patent Pending. All rights reserved.

This software is licensed under the terms of the Image-box-overlap licence
which allows for non-commercial use only, the full terms of which are made
available in the LICENSE file.
"""

import argparse


class OptionsBoxesTrain:
    def __init__(self):
        self.options = None
        self.parser = argparse.ArgumentParser(description='Predicting Visual Overlap of Images: Training Loop.')
        self.parser.add_argument('--name', default='debug', type=str,
                                 help='Name of the experiment (default: debug).')
        self.parser.add_argument('--dataset', type=str, default='megadepth',
                                 choices=['megadepth'], help='Parent dataset for scene (default: megadepth).')
        self.parser.add_argument('--dataset_json', default='data/dataset_jsons/megadepth/bigben.json',
                                 help='Path to dataset json files.')
        self.parser.add_argument('--log_path', default=None, type=str,
                                 help='Path to model and log files. If not specified it saves in current directory.')
        self.parser.add_argument('--model', default='resnet50', type=str,
                                 choices=['resnet18', 'resnet50', 'resnet101'],
                                 help='Backbone to use. (default: resnet50).')
        self.parser.add_argument('--box_ndim', default=32, type=int,
                                 help='Box embedding dimension (default: 32).')
        self.parser.add_argument('--input_hw', type=int, nargs='+', default=(256, 456),
                                 help='Network input height and width (default: (256,456).')
        self.parser.add_argument('--batch_size', type=int, default=32,
                                 help='Input batch size for training (default: 32).')
        self.parser.add_argument('--learning_rate', type=float, default=0.0001,
                                 help='Learning rate (default: 0.0001)')
        self.parser.add_argument('--num_epochs', type=int, default=20,
                                 help='Number of epochs to train (default: 20).')
        self.parser.add_argument('--log_frequency', type=int, default=250,
                                 help='Number of batches to run validation and log (default: 250).')
        self.parser.add_argument('--save_frequency', default=250, type=int,
                                 help='Save the model every N steps (default: 250).')
        self.parser.add_argument('--num_workers', type=int, default=8,
                                 help='Number of workers for data loading (default: 8).')
        self.parser.add_argument('--seed', default=42, type=int,
                                 help='Random seed for reproducibility.')
        self.parser.add_argument('--num_gpus', default=1, type=int,
                                 help='Number of gpus used for training (default: 1)')
        self.parser.add_argument('--backend', default='dp', type=str,
                                 help='Lightning distributed backend (default: dp)')

    def parse(self, *args, **kwargs):
        self.options = self.parser.parse_args(*args, **kwargs)
        return self.options


class OptionsBoxesTest:
    def __init__(self):
        self.options = None
        self.parser = argparse.ArgumentParser(description='Predicting Visual Overlap of Images: Test Loop')
        self.parser.add_argument('--dataset', type=str, default='megadepth',
                                 choices=['megadepth'], help='Parent dataset for scene.')
        self.parser.add_argument('--dataset_json', default='data/dataset_jsons/megadepth/bigben.json',
                                 help='Path to dataset json files.')
        self.parser.add_argument('--model_scene', default='bigben', type=str,
                                 choices=['bigben', 'notredame', 'florence', 'venice'], help='Choose the model to load')
        self.parser.add_argument('--model', default='resnet50', type=str,
                                 choices=['resnet18', 'resnet50', 'resnet101'], help='Backbone of saved model.')
        self.parser.add_argument('--box_ndim', default=32, type=int,
                                 help='Box embedding dimension (default: 32)')
        self.parser.add_argument('--input_hw', type=int, nargs='+', default=(256, 456),
                                 help='Network input height and width (default:(256,456)')
        self.parser.add_argument('--batch_size', type=int, default=8,
                                 help='Input batch size for training (default: 8)')
        self.parser.add_argument('--num_workers', type=int, default=8,
                                 help='Number of workers for data loading (default: 8)')

    def parse(self, *args, **kwargs):
        self.options = self.parser.parse_args(*args, **kwargs)
        return self.options