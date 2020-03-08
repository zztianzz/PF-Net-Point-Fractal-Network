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
from utils import PointLoss
from utils import distance_squre
from pointcapsnet_ae import PointCapsNet
import data_utils as d_utils


parser = argparse.ArgumentParser()
parser.add_argument('--infile',type = str, default = 'test_from_prn_to_capsule/crop3.csv')
parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
parser.add_argument('--n_epochs', type=int, default=300, help='number of epochs to train for')
parser.add_argument('--workers', type=int,default=2, help='number of data loading workers')
parser.add_argument('--prim_caps_size', type=int, default=1024, help='number of primary point caps')
parser.add_argument('--prim_vec_size', type=int, default=16, help='scale of primary point caps')
parser.add_argument('--latent_caps_size', type=int, default=64, help='number of latent caps')
parser.add_argument('--latent_vec_size', type=int, default=64, help='scale of latent caps')
parser.add_argument('--num_points', type=int, default=2048, help='input point set size')
parser.add_argument('--crop_point_num',type=int,default=512,help='number of crop points ')
parser.add_argument('--outf', type=str, default='tmp_checkpoints', help='output folder')
parser.add_argument('--model', type=str, default='Trained_Recon_Model_Capsule/3dCapsule110.pth', help='model path')
parser.add_argument('--dataset', type=str, default='shapenet_part', help='dataset: shapenet_part, shapenet_core13, shapenet_core55')
opt = parser.parse_args()
print(opt)

def distance_squre1(p1,p2):
    return (p1[0]-p2[0])**2+(p1[1]-p2[1])**2+(p1[2]-p2[2])**2 


test_dset = shapenet_part_loader.PartDataset( root='../dataset/shapenetcore_partanno_segmentation_benchmark_v0/',classification=True, class_choice='Car', npoints=opt.num_points, split='test')
test_dataloader = torch.utils.data.DataLoader(test_dset, batch_size=opt.batch_size,
                                         shuffle=True,num_workers = int(opt.workers))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
capsule_net = PointCapsNet(opt.prim_caps_size, opt.prim_vec_size, opt.latent_caps_size, opt.latent_caps_size, opt.num_points)

capsule_net.load_state_dict(torch.load(opt.model,map_location=lambda storage, location: storage)['state_dict'])      
print("Let's use", torch.cuda.device_count(), "GPUs!")
capsule_net.to(device)
capsule_net = torch.nn.DataParallel(capsule_net)
capsule_net.eval()

input_cropped1 = np.loadtxt(opt.infile,delimiter=',')
input_cropped1 = torch.FloatTensor(input_cropped1)
input_cropped1 = torch.unsqueeze(input_cropped1, 0)
#input_cropped1 =torch.FloatTensor(1,1, opt.num_points-opt.crop_point_num, 3)
#input_cropped = torch.unsqueeze(input_cropped, 0)
#p_center = torch.FloatTensor([0,0,0])
#distance_list = []
#for n in range(1792):
#    distance_list.append(distance_squre(input_cropped[0,n],p_center))
#distance_order = sorted(enumerate(distance_list), key  = lambda x:x[1])
#
#crop_num_list = []
#for num in range(1536):
#    crop_num_list.append(distance_order[num+256][0])
#indices = torch.LongTensor(crop_num_list)
#input_cropped1[0,0]=torch.index_select(input_cropped[0],0,indices)
#input_cropped1 = torch.squeeze(input_cropped1,1)

input_cropped1 = input_cropped1.transpose(2,1)
input_cropped1 = input_cropped1.to(device)

codewords,reconstruction = capsule_net(input_cropped1)
reconstruction_ = reconstruction.transpose(2, 1).contiguous()
input_cropped1 = input_cropped1.transpose(2,1).contiguous()

reconstruction_ =reconstruction_.cpu()
np_fake = reconstruction_[0].detach().numpy()

input_cropped1 = input_cropped1.cpu()
np_crop = input_cropped1[0].numpy() 

np.savetxt('test_from_prn_to_capsule/crop_capsule'+'.csv', np_crop, fmt = "%f,%f,%f")
np.savetxt('test_from_prn_to_capsule/fake_capsule'+'.csv', np_fake, fmt = "%f,%f,%f")



    
    