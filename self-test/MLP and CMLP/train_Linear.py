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
from Linear_model import Linear_autoencoder,CMLP_autoencoder
import data_utils as d_utils

parser = argparse.ArgumentParser()
parser.add_argument('--model_choose',type=int, default=0, help='0 test linear,1 test CMLP')
parser.add_argument('--batch_size', type=int, default=36, help='input batch size')
parser.add_argument('--n_epochs', type=int, default=201, help='number of epochs to train for')
parser.add_argument('--workers', type=int,default=2, help='number of data loading workers')
parser.add_argument('--num_points', type=int, default=2048, help='input point set size')
parser.add_argument('--crop_point_num',type=int,default=512,help='number of crop points ')
parser.add_argument('--outf', type=str, default='tmp_checkpoints', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--dataset', type=str, default='shapenet_part', help='dataset: shapenet_part, shapenet_core13, shapenet_core55')
opt = parser.parse_args()

resume_epoch=0
USE_CUDA = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if opt.model_choose == 0:
    Autoencoder = Linear_autoencoder(opt.num_points-opt.crop_point_num,opt.crop_point_num)
else:
    Autoencoder = CMLP_autoencoder(opt.num_points-opt.crop_point_num,opt.crop_point_num)

if opt.model != '':
    Autoencoder.load_state_dict(torch.load(opt.model,map_location=lambda storage, location: storage)['state_dict'])  
    resume_epoch = torch.load(opt.model)['epoch']
    
if USE_CUDA:       
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    Autoencoder.to(device)
    Autoencoder = torch.nn.DataParallel(Autoencoder)
    



dset = shapenet_part_loader.PartDataset( root='../../dataset/shapenetcore_partanno_segmentation_benchmark_v0/',classification=True, class_choice=None, npoints=opt.num_points, split='train')
assert dset
dataloader = torch.utils.data.DataLoader(dset, batch_size=opt.batch_size,
                                         shuffle=True,num_workers = int(opt.workers))
print(len(dataloader))

test_dset = shapenet_part_loader.PartDataset( root='../../dataset/shapenetcore_partanno_segmentation_benchmark_v0/',classification=True, class_choice=None, npoints=opt.num_points, split='test')
test_dataloader = torch.utils.data.DataLoader(test_dset, batch_size=opt.batch_size,
                                         shuffle=True,num_workers = int(opt.workers))

print(Autoencoder)
criterion_PointLoss = PointLoss().to(device)
optimizer = optim.Adam(Autoencoder.parameters(), lr=0.0001,betas=(0.9, 0.999))
schedulerG = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

for epoch in range(resume_epoch,opt.n_epochs): 
        
    for batch_id, data in enumerate(dataloader):  
        Autoencoder.train()
        real_point, target = data
        real_point = torch.unsqueeze(real_point,1)
        p_origin = [0,0,0]
        batch_size = real_point.size()[0]
        real_center = torch.FloatTensor(batch_size, 1, opt.crop_point_num, 3)
        input_cropped =torch.FloatTensor(batch_size, 1, opt.num_points-opt.crop_point_num, 3)
        choice =[torch.Tensor([1,0,0]),torch.Tensor([0,0,1]),torch.Tensor([1,0,1]),torch.Tensor([-1,0,0]),torch.Tensor([-1,1,0])]
        for m in range(batch_size):
            index = random.sample(choice,1)
            distance_list = []
            p_center = index[0]
            for n in range(opt.num_points):
                distance_list.append(distance_squre(real_point[m,0,n],p_center))
            distance_order = sorted(enumerate(distance_list), key  = lambda x:x[1])
            
            for sp in range(opt.crop_point_num):
                real_center.data[m,0,sp] = real_point[m,0,distance_order[sp][0]]
                
            crop_num_list = []
            for num in range(opt.num_points-opt.crop_point_num):
                crop_num_list.append(distance_order[num+opt.crop_point_num][0])
            indices = torch.LongTensor(crop_num_list)
            input_cropped[m,0]=torch.index_select(real_point[m,0],0,indices)
            

        real_center = torch.squeeze(real_center,1)
        input_cropped = torch.squeeze(input_cropped,1)
        real_center = Variable(real_center,requires_grad=True) 
        input_cropped = Variable(input_cropped,requires_grad=True) 
        real_center = real_center.to(device)
        input_cropped = input_cropped.to(device)        
        optimizer.zero_grad()
        reconstruction=Autoencoder(input_cropped)        
        errG = criterion_PointLoss(reconstruction,real_center)
        errG.backward()
        optimizer.step()
        print('[%d/%d][%d/%d]  Loss_G: %.4f '
              % (epoch, opt.n_epochs, batch_id, len(dataloader), errG))  
        f=open('Linear.txt','a')
        f.write('\n'+'[%d/%d][%d/%d]  Loss_G: %.4f '
              % (epoch, opt.n_epochs, batch_id, len(dataloader), errG))
        f.close()
    if epoch% 10 == 0:
        torch.save({'epoch':epoch+1,
                'state_dict':Autoencoder.module.state_dict()},
                'Trained_Recon_Model_Linear/Linear_ae'+str(epoch)+'.pth' )                
                    
    schedulerG.step()                
                    
