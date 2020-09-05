"""
Predicting Visual Overlap of Images Through Interpretable Non-Metric Box Embeddings.

Example of evaluating a trained model for surface overlap prediction.
If provided models are used it reproduces results on Table 1 of the paper.

Copyright Niantic 2020. Patent Pending. All rights reserved.

This software is licensed under the terms of the Image-box-overlap licence
which allows for non-commercial use only, the full terms of which are made
available in the LICENSE file.
"""

import torch
import os
from tqdm import tqdm
from .options import OptionsBoxesTest
from .model import ResnetEncoder
from .datasets import MegaDepthSurfacePairLoader
from .utils import box_overlap_soft, checkpoint_loader, download_model_if_doesnt_exist


def main():
    # Parse command line arguments
    opts = OptionsBoxesTest().parse()
    # Download the trained model if it doesn't exist to reproduce results.
    download_model_if_doesnt_exist(opts.model_scene)

    # If you train your own models using train.py, just change the path and use the .cpkt extension instead
    state_dict = checkpoint_loader(os.path.join('models', opts.model_scene, f'{opts.model_scene}.pth.tar'))
    test_loader = torch.utils.data.DataLoader(MegaDepthSurfacePairLoader(opts.dataset_json, mode='test'),
                                              batch_size=opts.batch_size,
                                              shuffle=False,
                                              num_workers=opts.num_workers,
                                              pin_memory=True)

    # Initializes the model. Gets number of Resnet layers from name
    model = ResnetEncoder(int(opts.model[-2:]), opts.box_ndim)
    model.load_state_dict(state_dict)

    if torch.cuda.is_available():
        model.to(torch.device('cuda'))
    model.eval()

    overlap_preds, overlap_gts = [], []

    print("Processing batches...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader)):
            if torch.cuda.is_available():
                for key, ipt in batch.items():
                    batch[key] = ipt.cuda()
            bx_c, bx_e = model(batch['images'][:, 0])
            by_c, by_e = model(batch['images'][:, 1])
            overlap_pred = box_overlap_soft(bx_c, bx_e, by_c, by_e)
            overlap_preds.append(overlap_pred)
            overlap_gts.append(batch['surface_overlap'].type_as(overlap_pred))

    overlap_gts = torch.cat(overlap_gts)
    overlap_preds = torch.cat(overlap_preds)
    diff = overlap_preds - overlap_gts
    diff_abs_pair = torch.abs(diff).reshape((int(len(diff) / 2), 2))

    print("Results:")
    print("RMSE: {:.3f}".format(torch.sqrt(torch.mean(torch.sum(diff_abs_pair ** 2, 1))).cpu().numpy()))
    print("L1 Norm: {:.3f}".format(torch.mean(torch.sum(diff_abs_pair, 1)).cpu().numpy()))
    print("Acc. < 0.1: {:.1f}%".format(100*torch.mean((torch.abs(diff) < 0.1).double()).cpu().numpy()))


if __name__ == '__main__':
    main()
