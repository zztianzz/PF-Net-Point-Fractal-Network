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
from utils import PointLoss,distance_squre,farthest_point_sample,index_points
from PCN_model import Autoencoder
import data_utils as d_utils

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=24, help='input batch size')
parser.add_argument('--n_epochs', type=int, default=101, help='number of epochs to train for')
parser.add_argument('--workers', type=int,default=2, help='number of data loading workers')
parser.add_argument('--num_inputs', type=int, default=1024, help='parital point numbers')
parser.add_argument('--num_coarses', type=int, default=1024, help='coarse point numbers')
parser.add_argument('--num_fines', type=int, default=2048, help='fine point numbers')
parser.add_argument('--grid_size', type=int, default=4, help='2d grid size')

parser.add_argument('--crop_point_num',type=int,default=512,help='number of crop points ')

parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--dataset', type=str, default='shapenet_part', help='dataset: shapenet_part, shapenet_core13, shapenet_core55')
opt = parser.parse_args()

resume_epoch=0
USE_CUDA = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
PCN = Autoencoder(opt.num_inputs,opt.num_coarses,opt.num_fines,opt.grid_size)

if opt.model != '':
    PCN.load_state_dict(torch.load(opt.model,map_location=lambda storage, location: storage)['state_dict'])  
    resume_epoch = torch.load(opt.model)['epoch']
    
if USE_CUDA:       
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    PCN.to(device)
    PCN = torch.nn.DataParallel(PCN)
    



dset = shapenet_part_loader.PartDataset( root='../dataset/shapenetcore_partanno_segmentation_benchmark_v0/',classification=True, class_choice=None, npoints=opt.num_fines, split='train')
assert dset
dataloader = torch.utils.data.DataLoader(dset, batch_size=opt.batch_size,
                                         shuffle=True,num_workers = int(opt.workers))


test_dset = shapenet_part_loader.PartDataset( root='../dataset/shapenetcore_partanno_segmentation_benchmark_v0/',classification=True, class_choice=None, npoints=opt.num_fines, split='test')
test_dataloader = torch.utils.data.DataLoader(test_dset, batch_size=opt.batch_size,
                                         shuffle=True,num_workers = int(opt.workers))

print(PCN)
criterion_PointLoss = PointLoss().to(device)



for epoch in range(resume_epoch,opt.n_epochs): 
    if epoch < 50:
        optimizer = optim.Adam(PCN.parameters(), lr=0.0001)
    elif epoch<150:
        optimizer = optim.Adam(PCN.parameters(), lr=0.00001)
    else:
        optimizer = optim.Adam(PCN.parameters(), lr=0.000001)
    
    if epoch<15:
        alpha = 0.01
    elif epoch<35:
        alpha = 0.1
    elif epoch<70:
        alpha = 0.5
    else:
        alpha = 1.0
        
    for batch_id, data in enumerate(dataloader):  
        PCN.train()
        real_point, target = data
        real_point = torch.unsqueeze(real_point,1)
        p_origin = [0,0,0]
        batch_size = real_point.size()[0]
        input_cropped =torch.FloatTensor(batch_size, 1, opt.num_fines-opt.crop_point_num, 3)
#        choice =[torch.Tensor([1,0,0]),torch.Tensor([0,0,1]),torch.Tensor([1,0,1])]
        choice = [torch.Tensor([1,0,0]),torch.Tensor([0,0,1]),torch.Tensor([1,0,1]),torch.Tensor([-1,0,0]),torch.Tensor([-1,1,0])]
        for m in range(batch_size):
            index = random.sample(choice,1)
            distance_list = []
            p_center = index[0]
            for n in range(opt.num_fines):
                distance_list.append(distance_squre(real_point[m,0,n],p_center))
            distance_order = sorted(enumerate(distance_list), key  = lambda x:x[1])
            
            crop_num_list = []
            for num in range(opt.num_fines-opt.crop_point_num):
                crop_num_list.append(distance_order[num+opt.crop_point_num][0])
            indices = torch.LongTensor(crop_num_list)
            input_cropped[m,0]=torch.index_select(real_point[m,0],0,indices)
            
        real_point = torch.squeeze(real_point,1) #BX2048X3
        input_cropped = torch.squeeze(input_cropped,1) #BX(2048-512)X3
        
        real_key_point_index =farthest_point_sample(real_point,opt.num_coarses,False)
        real_key_point = index_points(real_point,real_key_point_index) #BX1024X3
        
        input_key_cropped_index = farthest_point_sample(input_cropped,opt.num_inputs,False)
        input_key_cropped = index_points(input_cropped,input_key_cropped_index) #BX1024X3
        
        
        real_point = Variable(real_point,requires_grad=True)  
        real_key_point = Variable(real_key_point,requires_grad=True) 
        input_key_cropped = Variable(input_key_cropped,requires_grad=True)
        
        
        real_point = real_point.to(device)
        real_key_point = real_key_point.to(device)
        input_key_cropped = input_key_cropped.to(device)        
        optimizer.zero_grad()
        
        
        coarses,fine=PCN(input_key_cropped) 
       
        errG = criterion_PointLoss(fine,real_point)+alpha*criterion_PointLoss(coarses,real_key_point)
        errG.backward()
        optimizer.step()
        print('[%d/%d][%d/%d]  Loss_G: %.4f '
              % (epoch, opt.n_epochs, batch_id, len(dataloader), errG))  
        f=open('loss_PCN.txt','a')
        f.write('\n'+'[%d/%d][%d/%d]  Loss_G: %.4f '
              % (epoch, opt.n_epochs, batch_id, len(dataloader), errG))
        

        f.close()
    if epoch% 10 == 0:
        torch.save({'epoch':epoch+1,
                'state_dict':PCN.module.state_dict()},
                'Trained_Recon_Model_PCN/pcn'+str(epoch)+'.pth' )                
                    
                    
                    