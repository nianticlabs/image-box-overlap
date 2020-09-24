"""
Predicting Visual Overlap of Images Through Interpretable Non-Metric Box Embeddings.

Training loop.

Copyright Niantic 2020. Patent Pending. All rights reserved.

This software is licensed under the terms of the Image-box-overlap licence
which allows for non-commercial use only, the full terms of which are made
available in the LICENSE file.
"""

import os
import torch
import pytorch_lightning as pl
from .options import OptionsBoxesTrain
from .datasets import MegaDepthSurfacePairLoader
from .trainer import BoxSurfaceOverlap


def main():
    # Parse command line arguments
    opts = OptionsBoxesTrain().parse()
    # Random seeds
    pl.seed_everything(opts.seed)
    # Setup logging
    path_logs = os.path.join(opts.log_path, opts.name) \
        if opts.log_path is not None else os.path.join(os.getcwd(), opts.name)

    if not os.path.exists(path_logs):
        os.makedirs(path_logs)

    logger = pl.loggers.TensorBoardLogger(path_logs, name=opts.name)

    # Dataloader initialization
    if opts.dataset.lower() == 'megadepth':
        train_dataloader = torch.utils.data.DataLoader(
            MegaDepthSurfacePairLoader(opts.dataset_json, mode='train'),
            batch_size=opts.batch_size,
            shuffle=True, num_workers=opts.num_workers,
            pin_memory=True)
        val_dataloader = torch.utils.data.DataLoader(
            MegaDepthSurfacePairLoader(opts.dataset_json, mode='val'),
            batch_size=opts.batch_size,
            shuffle=False, num_workers=opts.num_workers,
            pin_memory=True)
    else:
        raise NotImplemented(f"Dataset {opts.dataset} not implemented!")

    # Initialize network
    model = BoxSurfaceOverlap(opts)

    # Initialize training loop
    trainer = pl.Trainer(default_save_path=path_logs,
                         logger=logger,
                         gpus=opts.num_gpus,
                         val_check_interval=opts.log_frequency,
                         distributed_backend=opts.backend,
                         deterministic=True,
                         fast_dev_run=False,
                         max_epochs=opts.num_epochs)

    # Train
    trainer.fit(model, train_dataloader=train_dataloader, val_dataloaders=val_dataloader)


if __name__ == '__main__':
    main()
