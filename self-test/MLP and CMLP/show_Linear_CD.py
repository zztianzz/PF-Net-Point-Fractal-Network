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
from utils import PointLoss_test
from utils import distance_squre
from Linear_model import Linear_autoencoder,CMLP_autoencoder
import data_utils as d_utils

parser = argparse.ArgumentParser()
parser.add_argument('--model_choose',type=int, default=0, help='0 test linear,1 test CMLP')
parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
parser.add_argument('--n_epochs', type=int, default=101, help='number of epochs to train for')
parser.add_argument('--workers', type=int,default=2, help='number of data loading workers')
parser.add_argument('--num_points', type=int, default=2048, help='input point set size')
parser.add_argument('--crop_point_num',type=int,default=512,help='number of crop points ')
parser.add_argument('--outf', type=str, default='tmp_checkpoints', help='output folder')
parser.add_argument('--model', type=str, default='Trained_Recon_Model_Linear/Linear_ae110.pth', help='model path')
parser.add_argument('--dataset', type=str, default='shapenet_part', help='dataset: shapenet_part, shapenet_core13, shapenet_core55')
opt = parser.parse_args()


test_dset = shapenet_part_loader.PartDataset( root='../../dataset/shapenetcore_partanno_segmentation_benchmark_v0/',classification=True, class_choice='Chair', npoints=opt.num_points, split='test')
test_dataloader = torch.utils.data.DataLoader(test_dset, batch_size=opt.batch_size,
                                         shuffle=False,num_workers = int(opt.workers))
length = len(test_dataloader)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if opt.model_choose == 0:
    Autoencoder = Linear_autoencoder(opt.num_points-opt.crop_point_num,opt.crop_point_num)
else:
    Autoencoder = CMLP_autoencoder(opt.num_points-opt.crop_point_num,opt.crop_point_num)
Autoencoder.load_state_dict(torch.load(opt.model,map_location=lambda storage, location: storage)['state_dict'])      
print("Let's use", torch.cuda.device_count(), "GPUs!")
Autoencoder.to(device)
Autoencoder = torch.nn.DataParallel(Autoencoder)
Autoencoder.eval()


criterion_PointLoss = PointLoss_test().to(device)
errG_min = 100
n = 0
CD = 0
Gt_Pre =0
Pre_Gt = 0
IDX = 1
for i, data in enumerate(test_dataloader, 0):
        
    real_point, target = data
    real_point = torch.unsqueeze(real_point, 1)
    input_cropped =torch.FloatTensor(opt.batch_size, 1, opt.num_points-opt.crop_point_num, 3)
    real_center = torch.FloatTensor(opt.batch_size, 1, opt.crop_point_num, 3)
    fake_center = torch.FloatTensor(opt.batch_size, 1, opt.crop_point_num, 3)
    batch_size = real_point.size()[0]
    p_origin = [0,0,0]
    choice =[torch.Tensor([1,0,0]),torch.Tensor([0,0,1]),torch.Tensor([1,0,1]),torch.Tensor([-1,0,0]),torch.Tensor([-1,1,0])]   
#    points = [x for x in range(0,opt.pnum-1)]
#    choice =random.sample(points,5)
    index = choice[IDX-1]#random.sample(choice,1)
    IDX  = IDX+1
    if IDX%5 == 0:
        IDX = 0
    distance_list = []
#    p_center  = real_point[0,0,index]
    p_center = index
    for num in range(opt.num_points):
        distance_list.append(distance_squre(real_point[0,0,num],p_center))
    distance_order = sorted(enumerate(distance_list), key = lambda x:x[1])
    
    for sp in range(opt.crop_point_num):
        real_center.data[0,0,sp] = real_point[0,0,distance_order[sp][0]]
    real_center = torch.squeeze(real_center,1)  
    
    crop_num_list = []
    for num in range(opt.num_points-opt.crop_point_num):
        crop_num_list.append(distance_order[num+opt.crop_point_num][0])
    indices = torch.LongTensor(crop_num_list)
    input_cropped[0,0]=torch.index_select(real_point[0,0],0,indices)
    
    
    real_point = torch.squeeze(real_point,1)
    input_cropped = torch.squeeze(input_cropped,1)
    real_point = real_point.to(device)
    input_cropped = input_cropped.to(device)    
    reconstruction=Autoencoder(input_cropped) 
    
#    reconstruction = reconstruction.cpu()
#    distance_list_fake=[]
#    for num_fake in range(opt.num_points):
#        distance_list_fake.append(distance_squre(reconstruction[0,num_fake],p_center))
#    distance_order = sorted(enumerate(distance_list_fake), key = lambda x:x[1])

#    for sp_fake in range(opt.crop_point_num):
#        fake_center.data[0,0,sp_fake] = reconstruction[0,distance_order[sp_fake][0]]
#    fake_center = torch.squeeze(fake_center,1)    
    
    reconstruction = reconstruction.to(device)
    real_center = real_center.to(device)
#    real_center_key =real_center_key.cuda()
    dist_all, dist1, dist2 = criterion_PointLoss(reconstruction,real_center)#change this to show cd on whole shape or part shape
    
    
    dist_all=dist_all.cpu().detach().numpy()
    dist1 =dist1.cpu().detach().numpy()
    dist2 = dist2.cpu().detach().numpy()
    CD = CD + dist_all/length
    Gt_Pre = Gt_Pre + dist1/length
    Pre_Gt = Pre_Gt + dist2/length
    print(CD,Gt_Pre,Pre_Gt)
print(CD,Gt_Pre,Pre_Gt)
print("CD:{} , Gt_Pre:{} , Pre_Gt:{}".format(float(CD),float(Gt_Pre),float(Pre_Gt)))
print(length)    
    

#
    
    
    
    
