#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import torch
import torch.nn.parallel
from torch.autograd import Variable
import torch.optim as optim
import random
import sys
import os
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append('../..')
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, 'models')))
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, 'dataloaders')))
import shapenet_part_loader
from utils import PointLoss
from utils import distance_squre
from LGAN_model import LGAN_autoencoder
import data_utils as d_utils

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
parser.add_argument('--infile',type = str, default = 'test_from_prn_to_LGAN/crop3.csv')
parser.add_argument('--n_epochs', type=int, default=101, help='number of epochs to train for')
parser.add_argument('--workers', type=int,default=2, help='number of data loading workers')
parser.add_argument('--num_points', type=int, default=2048, help='input point set size')
parser.add_argument('--crop_point_num',type=int,default=512,help='number of crop points ')
parser.add_argument('--outf', type=str, default='tmp_checkpoints', help='output folder')
parser.add_argument('--model', type=str, default='Trained_Recon_Model_LGAN/LGAN_ae55.pth', help='model path')
parser.add_argument('--dataset', type=str, default='shapenet_part', help='dataset: shapenet_part, shapenet_core13, shapenet_core55')
opt = parser.parse_args()



test_dset = shapenet_part_loader.PartDataset( root='../../dataset/shapenetcore_partanno_segmentation_benchmark_v0/',classification=True, class_choice='Airplane', npoints=opt.num_points, split='test')
test_dataloader = torch.utils.data.DataLoader(test_dset, batch_size=opt.batch_size,
                                         shuffle=True,num_workers = int(opt.workers))


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LGAN_autoencoder = LGAN_autoencoder(opt.num_points-opt.crop_point_num,opt.num_points)
LGAN_autoencoder.load_state_dict(torch.load(opt.model,map_location=lambda storage, location: storage)['state_dict'])      
print("Let's use", torch.cuda.device_count(), "GPUs!")
LGAN_autoencoder.to(device)
LGAN_autoencoder = torch.nn.DataParallel(LGAN_autoencoder)
LGAN_autoencoder.eval()


input_cropped1 = np.loadtxt(opt.infile,delimiter=',')
input_cropped1 = torch.FloatTensor(input_cropped1)
input_cropped1 = torch.unsqueeze(input_cropped1, 0)
input_cropped1 = input_cropped1.to(device)
reconstruction=LGAN_autoencoder(input_cropped1) 

reconstruction = reconstruction.cpu()
np_fake = reconstruction[0].detach().numpy()
input_cropped1 = input_cropped1.cpu()
np_crop = input_cropped1[0].numpy()

np.savetxt('test_from_prn_to_LGAN/crop_LGAN'+'.csv', np_crop, fmt = "%f,%f,%f")
np.savetxt('test_from_prn_to_LGAN/fake_LGAN'+'.csv', np_fake, fmt = "%f,%f,%f")


    
    
    
    
