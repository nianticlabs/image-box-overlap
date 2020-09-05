"""
Predicting Visual Overlap of Images Through Interpretable Non-Metric Box Embeddings.

MegaDepth dataset dataloaders.

Copyright Niantic 2020. Patent Pending. All rights reserved.

This software is licensed under the terms of the Image-box-overlap licence
which allows for non-commercial use only, the full terms of which are made
available in the LICENSE file.
"""

import os
import numpy as np
import torch.utils.data
import json
import errno
from PIL import Image
from torchvision import transforms
from .dataset_generator.utils import pil_loader, params2matrix, quat2mat


class MegaDepthSurfacePairLoader(torch.utils.data.Dataset):
    """
    Loader for pairs of images and their respective normalized surface overlap (NSO)
    """

    def __init__(self, dataset_json_path, mode, image_hw=None, loader=pil_loader):
        super(MegaDepthSurfacePairLoader, self).__init__()
        if image_hw is None:
            image_hw = [256, 456]
        self.height, self.width = image_hw
        self.resizer = transforms.Resize((self.height, self.width), interpolation=Image.ANTIALIAS)
        self.to_tensor = transforms.ToTensor()
        self.loader = loader
        image_0, image_1, surface_overlaps = [], [], []
        num_pairs = 0

        with open(dataset_json_path, 'r') as f:
            dataset_json = json.load(f)

        for entry in dataset_json:
            if mode == 'train':
                image_list = entry['train_file']
            elif mode == 'val':
                image_list = entry['val_file']
            elif mode == 'test':
                image_list = entry['test_file']
            else:
                raise NotImplementedError("No valid split chosen.")

            scene_path = os.path.join(entry['path_sfm'], entry['scene'], 'images')

            with open(image_list, 'r') as f:
                for line in f.readlines():
                    image_0_path = os.path.join(scene_path, line.split()[0])
                    image_1_path = os.path.join(scene_path, line.split()[1])

                    if os.path.isfile(image_0_path):
                        image_0.append(image_0_path)
                    else:
                        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), image_0_path)

                    if os.path.isfile(image_1_path):
                        image_1.append(image_1_path)
                    else:
                        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), image_1_path)

                    surface_overlaps.append(np.array(line.split()[2]).astype(np.float))
                    num_pairs += 1

        self.image_0_paths = image_0
        self.image_1_paths = image_1
        self.surface_overlaps = surface_overlaps
        self.num_pairs = num_pairs

    def __getitem__(self, index):

        image_0_original = self.loader(self.image_0_paths[index])
        image_1_original = self.loader(self.image_1_paths[index])

        image_0 = self.to_tensor(self.resizer(image_0_original))
        image_1 = self.to_tensor(self.resizer(image_1_original))

        batch = {
            'images': np.stack((image_0, image_1), 0),
            'surface_overlap': self.surface_overlaps[index]
        }
        return batch

    def __len__(self):
        return self.num_pairs


class MegaDepthImageLoader(torch.utils.data.Dataset):
    """
    Loader for single MegaDepth images. Apart from color and depth images paths
    it also returns camera pose and camera calibration parameters.
    """

    def __init__(self, dataset_json_path, mode, image_hw=None,
                 loader=pil_loader):
        super(MegaDepthImageLoader, self).__init__()

        if image_hw is None:
            image_hw = [256, 456]

        self.height, self.width = image_hw
        self.resizer = transforms.Resize((self.height, self.width))
        self.to_tensor = transforms.ToTensor()
        self.loader = loader
        self.mode = mode

        with open(dataset_json_path, 'r') as f:
            dataset_json = json.load(f)

        image_ids, image_paths, depth_paths, aligned_paths = [], [], [], []
        camera_poses, camera_calibs = [], []

        for entry in dataset_json:
            scene_path = os.path.join(entry['path_sfm'], entry['scene'], 'images')
            #  if you wish to use this loader to load single images and input in the network, for example to perform
            #  image retrieval, just add an additional mode with the appropriate json entry, everything else is ready.
            if self.mode == "data_generation":
                files_pointer = entry['list_images_with_depth']
            else:
                raise NotImplementedError("No valid mode chosen.")

            image_ids.extend([line.split()[0] for line in open(files_pointer)])

            image_paths.extend([os.path.join(scene_path, line.split()[0]) for line in open(files_pointer)])
            for path_i in image_paths:
                if not os.path.isfile(path_i):
                    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path_i)

            depth_paths.extend(
                [os.path.join(entry['path_depth'], entry['scene'], 'dense0', 'depths', f"{line.split('.')[0]}.h5") for
                 line in open(files_pointer)])

            for path_d in depth_paths:
                if not os.path.isfile(path_d):
                    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path_d)

            aligned_paths.extend(
                [os.path.join(entry['path_depth'], entry['scene'], 'dense0', 'imgs', f"{line.split('.')[0]}.jpg") for
                 line in open(files_pointer)])

            for path_a in image_paths:
                if not os.path.isfile(path_a):
                    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path_a)

            camera_params, annotated_sizes = {}, {}
            with open(os.path.join(entry['path_sfm'], entry['scene'], 'sparse', 'manhattan', '0', 'cameras.txt'),
                      'r') as f:
                lines = f.readlines()[3:]
                for line in lines:
                    tmp_str = line.split(' ')
                    # CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
                    camera_id = tmp_str[0]
                    camera_params[camera_id] = [float(params) for params in tmp_str[4:]]
                    annotated_sizes[camera_id] = tuple([int(tmp_str[2]), int(tmp_str[3])])

            reader = open(os.path.join(entry['path_sfm'], entry['scene'], 'sparse', 'manhattan', '0', 'images.txt'),
                          'r')

            poses_dict, cameras_dict = {}, {}
            for i, line in enumerate(reader):
                # image list with two lines of data per image:
                #   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
                #   POINTS2D[] as (X, Y, POINT3D_ID)
                if i % 2 == 0 and i > 3:
                    line = line.split()
                    tmp_img_name = line[-1]
                    tmp_camera_id = line[-2]
                    tmp_pose = np.array(line[1:8]).astype(np.float)
                    # rearrange quaternion from [qw qx qy qz] to [qx qy qz qw]
                    tmp_quat = np.concatenate((tmp_pose[1:4], tmp_pose[0]), axis=None)
                    tmp_t = tmp_pose[4:]
                    absolute_path_to_image = os.path.join(entry['path_sfm'], entry['scene'], 'images', tmp_img_name)
                    poses_dict[absolute_path_to_image] = quat2mat(tmp_quat, tmp_t)
                    cameras_dict[absolute_path_to_image] = tmp_camera_id
            reader.close()

            for i in range(len(image_paths)):
                camera_poses.append(poses_dict[image_paths[i]])
                # color image aligned to depth may have different size as original color image. See supplementary.
                tmp_camera = cameras_dict[image_paths[i]]
                tmp_params = camera_params[tmp_camera]
                tmp_anno_size = annotated_sizes[tmp_camera]
                image_aligned = Image.open(os.path.join(aligned_paths[i]))
                adjusted_calib = params2matrix(tmp_params, image_aligned.size, tmp_anno_size)
                camera_calibs.append(adjusted_calib)

        self.image_ids = image_ids
        self.image_paths = image_paths
        self.depth_paths = depth_paths
        self.camera_poses = camera_poses
        self.camera_calibs = camera_calibs

    def __getitem__(self, index):
        if self.mode == 'data_generation':
            dict_batch = {
                'image_id': self.image_ids[index],
                'image_path': self.image_paths[index],
                'depth_path': self.depth_paths[index],
                'camera_calib': self.camera_calibs[index],
                'camera_pose': self.camera_poses[index],
            }
        else:
            dict_batch = {
                'image': self.to_tensor(self.resizer(self.loader(self.image_paths[index]))),
                'image_id': self.image_ids[index],
                'image_path': self.image_paths[index],
                'depth_path': self.depth_paths[index],
                'camera_calib': self.camera_calibs[index],
                'camera_pose': self.camera_poses[index],
            }
        return dict_batch

    def __len__(self):
        return len(self.image_paths)
