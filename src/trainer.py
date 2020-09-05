"""
Predicting Visual Overlap of Images Through Interpretable Non-Metric Box Embeddings.

Trainer definition within PyTorch Lightning's framework.

Copyright Niantic 2020. Patent Pending. All rights reserved.

This software is licensed under the terms of the Image-box-overlap licence
which allows for non-commercial use only, the full terms of which are made
available in the LICENSE file.
"""

import torch
import pytorch_lightning as pl
from .utils import box_overlap_soft
from .model import ResnetEncoder


class BoxSurfaceOverlap(pl.LightningModule):

    def __init__(self, hparams):
        super(BoxSurfaceOverlap, self).__init__()
        self.hparams = hparams
        self.box_ndim = hparams.box_ndim
        # Get number of Resnet layers from name
        self.model = ResnetEncoder(int(self.hparams.model[-2:]), self.box_ndim)

    def forward(self, x):
        return self.model.forward(x)

    def _shared_eval(self, batch):
        bx_center, bx_extent = self.forward((batch['images'][:, 0]))
        by_center, by_extent = self.forward((batch['images'][:, 1]))

        overlap_pred = box_overlap_soft(bx_center, bx_extent, by_center, by_extent)
        # also moves tensor to overlap_pred's device
        overlap_gt = batch['surface_overlap'].type_as(overlap_pred)

        loss = torch.nn.MSELoss(reduction='mean')
        loss = loss(overlap_pred, overlap_gt)

        accuracy = (torch.abs((overlap_pred - overlap_gt)) < 0.1).sum().double() / len(overlap_gt)

        return loss, accuracy

    def training_step(self, batch, batch_idx):
        loss, accuracy = self._shared_eval(batch)
        return {'loss': loss,
                'progress_bar': {'accuracy': accuracy},
                'log': {'train_loss': loss, 'train_accuracy': accuracy}}

    def validation_step(self, batch, batch_idx):
        loss, accuracy = self._shared_eval(batch)
        return {'val_loss': loss, 'val_accuracy': accuracy}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_accuracy'] for x in outputs]).mean()
        return {'val_loss': avg_loss, 'log': {'val_loss': avg_loss, 'val_accuracy': avg_acc}}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        return [optimizer], [scheduler]