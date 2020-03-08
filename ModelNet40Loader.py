#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import torch
import torch.utils.data as data
import numpy as np
import os
import h5py
import subprocess
import shlex
import utils
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _get_data_files(list_filename):
    with open(list_filename) as f:
        return [line.rstrip()[5:] for line in f]


def _load_data_file(name):
    f = h5py.File(name)
    data = f["data"][:]
    label = f["label"][:]
    return data, label

class ModelNet40Cls(data.Dataset):
    def __init__(self, num_points, transforms=None, train=True, download=True):
        super().__init__()

        self.transforms = transforms

        self.folder = "modelnet40_ply_hdf5_2048"
        self.data_dir = os.path.join(BASE_DIR, self.folder)
        self.url = "https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip"

        if download and not os.path.exists(self.data_dir):
            zipfile = os.path.join(BASE_DIR, os.path.basename(self.url))
            subprocess.check_call(
                shlex.split("curl {} -o {}".format(self.url, zipfile))
            )

            subprocess.check_call(
                shlex.split("unzip {} -d {}".format(zipfile, BASE_DIR))
            )

            subprocess.check_call(shlex.split("rm {}".format(zipfile)))

        self.train, self.num_points = train, num_points
        if self.train:
            self.files = _get_data_files(os.path.join(self.data_dir, "train_files.txt"))
        else:
            self.files = _get_data_files(os.path.join(self.data_dir, "test_files.txt"))

        point_list, label_list = [], []
        for f in self.files:
            points, labels = _load_data_file(os.path.join(BASE_DIR, f))
            point_list.append(points)
            label_list.append(labels)

        self.points = np.concatenate(point_list, 0)
        self.labels = np.concatenate(label_list, 0)

        self.randomize()

    def __getitem__(self, idx):
        pt_idxs = np.arange(0, self.actual_number_of_points)
        np.random.shuffle(pt_idxs)

        current_points = self.points[idx, pt_idxs].copy()
        label = torch.from_numpy(self.labels[idx]).type(torch.LongTensor)

        if self.transforms is not None:
            current_points = self.transforms(current_points)

        return current_points, label

    def __len__(self):
        return self.points.shape[0]

    def set_num_points(self, pts):
        self.num_points = pts
        self.actual_number_of_points = pts

    def randomize(self):
        self.actual_number_of_points = min(
            max(self.num_points, 1),
            self.points.shape[1],
        )


if __name__ == "__main__":
    from torchvision import transforms
    import data_utils as d_utils

    transforms = transforms.Compose(
        [
            d_utils.PointcloudToTensor(),
#            d_utils.PointcloudRotate(axis=np.array([1, 0, 0])),
#            d_utils.PointcloudScale(),
#            d_utils.PointcloudTranslate(),
#            d_utils.PointcloudJitter(),
        ]
    )
    dset = ModelNet40Cls(1024, train=True, transforms=transforms)   
    print(dset[0][1])
    print(len(dset))
    dloader = torch.utils.data.DataLoader(dset, batch_size=64, shuffle=True)

    for i, Data in enumerate(dloader, 0):
        real_point, target = Data
        print('1')
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        real_point = real_point.to(device)
        real_point2_idx = utils.farthest_point_sample(real_point,512)
        real_point2 = utils.index_points(real_point,real_point2_idx)
        real_point3_idx = utils.farthest_point_sample(real_point,256)
        real_point3 = utils.index_points(real_point,real_point3_idx)
        
#        model1 = real_point[0].numpy()
#        model2 = real_point2[0].numpy()
#        model3 = real_point3[0].numpy()
#        
#        np.savetxt('test-examples/model'+'.txt', model1, fmt = "%f %f %f")
#        np.savetxt('test-examples/model2'+'.txt', model2, fmt = "%f %f %f")
#        np.savetxt('test-examples/model3'+'.txt', model3, fmt = "%f %f %f")