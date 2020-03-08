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
sys.path.append('..')
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, 'models')))
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, 'dataloaders')))
import shapenet_part_loader
from utils import PointLoss,distance_squre,farthest_point_sample,index_points
from PCN_model import Autoencoder
import data_utils as d_utils

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=24, help='input batch size')
parser.add_argument('--infile',type = str, default = 'test_from_prn_to_PCN/crop3.csv')
parser.add_argument('--n_epochs', type=int, default=101, help='number of epochs to train for')
parser.add_argument('--workers', type=int,default=2, help='number of data loading workers')
parser.add_argument('--num_inputs', type=int, default=1024, help='parital point numbers')
parser.add_argument('--num_coarses', type=int, default=1024, help='coarse point numbers')
parser.add_argument('--num_fines', type=int, default=2048, help='fine point numbers')
parser.add_argument('--grid_size', type=int, default=4, help='2d grid size')

parser.add_argument('--crop_point_num',type=int,default=512,help='number of crop points ')

parser.add_argument('--model', type=str, default='Trained_Recon_Model_PCN_5views/pcn100.pth', help='model path')
parser.add_argument('--dataset', type=str, default='shapenet_part', help='dataset: shapenet_part, shapenet_core13, shapenet_core55')
opt = parser.parse_args()



test_dset = shapenet_part_loader.PartDataset( root='../dataset/shapenetcore_partanno_segmentation_benchmark_v0/',classification=True, class_choice='Airplane', npoints=opt.num_fines, split='test')
test_dataloader = torch.utils.data.DataLoader(test_dset, batch_size=opt.batch_size,
                                         shuffle=True,num_workers = int(opt.workers))


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
PCN = Autoencoder(opt.num_inputs,opt.num_coarses,opt.num_fines,opt.grid_size)
PCN.load_state_dict(torch.load(opt.model,map_location=lambda storage, location: storage)['state_dict'])      
print("Let's use", torch.cuda.device_count(), "GPUs!")
PCN.to(device)
PCN = torch.nn.DataParallel(PCN)
PCN.eval()


input_cropped1 = np.loadtxt(opt.infile,delimiter=',')
input_cropped1 = torch.FloatTensor(input_cropped1)
input_cropped1 = torch.unsqueeze(input_cropped1, 0)
input_cropped1 = input_cropped1.to(device)
input_key_cropped_index = farthest_point_sample(input_cropped1,opt.num_inputs,RAN=False)
input_key_cropped = index_points(input_cropped1,input_key_cropped_index) #BX1024X3

coarses,fine=PCN(input_key_cropped) 
fine = fine.cpu()
np_fake = fine[0].detach().numpy()
input_cropped1 = input_cropped1.cpu()
np_crop = input_cropped1[0].numpy()

np.savetxt('test_from_prn_to_PCN/crop_PCN'+'.csv', np_crop, fmt = "%f,%f,%f")
np.savetxt('test_from_prn_to_PCN/fake_PCN'+'.csv', np_fake, fmt = "%f,%f,%f")


    
    
    
    
