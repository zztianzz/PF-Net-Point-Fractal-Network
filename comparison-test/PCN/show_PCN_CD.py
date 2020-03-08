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
from utils import PointLoss_test,distance_squre,farthest_point_sample,index_points
from PCN_model import Autoencoder
import data_utils as d_utils

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
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



test_dset = shapenet_part_loader.PartDataset( root='../dataset/shapenetcore_partanno_segmentation_benchmark_v0/',classification=True, class_choice='Table', npoints=opt.num_fines, split='test')
test_dataloader = torch.utils.data.DataLoader(test_dset, batch_size=opt.batch_size,
                                         shuffle=False,num_workers = int(opt.workers))
length = len(test_dataloader)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
PCN = Autoencoder(opt.num_inputs,opt.num_coarses,opt.num_fines,opt.grid_size)
PCN.load_state_dict(torch.load(opt.model,map_location=lambda storage, location: storage)['state_dict'])      
print("Let's use", torch.cuda.device_count(), "GPUs!")
PCN.to(device)
PCN = torch.nn.DataParallel(PCN)
PCN.eval()


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
    input_cropped =torch.FloatTensor(opt.batch_size, 1, opt.num_fines-opt.crop_point_num, 3)
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
    for num in range(opt.num_fines):
        distance_list.append(distance_squre(real_point[0,0,num],p_center))
    distance_order = sorted(enumerate(distance_list), key = lambda x:x[1])
    
    for sp in range(opt.crop_point_num):
        real_center.data[0,0,sp] = real_point[0,0,distance_order[sp][0]]
    real_center = torch.squeeze(real_center,1) 
    real_center = real_center.to(device)
    
    crop_num_list = []
    for num in range(opt.num_fines-opt.crop_point_num):
        crop_num_list.append(distance_order[num+opt.crop_point_num][0])
    indices = torch.LongTensor(crop_num_list)
    input_cropped[0,0]=torch.index_select(real_point[0,0],0,indices)
    
    
    real_point = torch.squeeze(real_point,1)
    input_cropped = torch.squeeze(input_cropped,1)
    
    input_key_cropped_index = farthest_point_sample(input_cropped,opt.num_inputs,RAN=False)
    input_key_cropped = index_points(input_cropped,input_key_cropped_index) #BX1024X3
    
    
    input_key_cropped = input_key_cropped.to(device)    
    coarses,fine=PCN(input_key_cropped) 
   
    fine = fine.cpu()
#    real_point  = real_point.cpu()
#    real_center = real_center.cpu()
    
    distance_list_fake=[]
    for num_fake in range(opt.num_fines):
        distance_list_fake.append(distance_squre(fine[0,num_fake],p_center))
    distance_order = sorted(enumerate(distance_list_fake), key = lambda x:x[1])
    
    for sp_fake in range(opt.crop_point_num):
        fake_center.data[0,0,sp_fake] = fine[0,distance_order[sp_fake][0]]
    fake_center = torch.squeeze(fake_center,1)
    fake_center = fake_center.to(device)
    real_point = real_point.to(device)
    fine = fine.to(device)
#    fake_center = fake_center.cpu()
    
    dist_all, dist1, dist2 = criterion_PointLoss(fine,real_point) #change this to show cd on whole shape or part shape
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
    
    
    
    
