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
parser.add_argument('--batch_size', type=int, default=20, help='input batch size')
parser.add_argument('--n_epochs', type=int, default=101, help='number of epochs to train for')
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
LGAN_autoencoder = LGAN_autoencoder(opt.num_points-opt.crop_point_num,opt.num_points)

if opt.model != '':
    LGAN_autoencoder.load_state_dict(torch.load(opt.model,map_location=lambda storage, location: storage)['state_dict'])  
    resume_epoch = torch.load(opt.model)['epoch']
    
if USE_CUDA:       
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    LGAN_autoencoder.to(device)
    LGAN_autoencoder = torch.nn.DataParallel(LGAN_autoencoder)
    



dset = shapenet_part_loader.PartDataset( root='../../dataset/shapenetcore_partanno_segmentation_benchmark_v0/',classification=True, class_choice=None, npoints=opt.num_points, split='train')
assert dset
dataloader = torch.utils.data.DataLoader(dset, batch_size=opt.batch_size,
                                         shuffle=True,num_workers = int(opt.workers))


test_dset = shapenet_part_loader.PartDataset( root='../../dataset/shapenetcore_partanno_segmentation_benchmark_v0/',classification=True, class_choice=None, npoints=opt.num_points, split='test')
test_dataloader = torch.utils.data.DataLoader(test_dset, batch_size=opt.batch_size,
                                         shuffle=True,num_workers = int(opt.workers))

print(LGAN_autoencoder)
criterion_PointLoss = PointLoss().to(device)
optimizer = optim.Adam(LGAN_autoencoder.parameters(), lr=0.0005,betas=(0.9, 0.999))


for epoch in range(resume_epoch,opt.n_epochs): 
        
    for batch_id, data in enumerate(dataloader):  
        LGAN_autoencoder.train()
        real_point, target = data
        real_point = torch.unsqueeze(real_point,1)
        p_origin = [0,0,0]
        batch_size = real_point.size()[0]
        input_cropped =torch.FloatTensor(batch_size, 1, opt.num_points-opt.crop_point_num, 3)
        choice =[torch.Tensor([1,0,0]),torch.Tensor([0,0,1]),torch.Tensor([1,0,1]),torch.Tensor([-1,0,0]),torch.Tensor([-1,1,0])]
        for m in range(batch_size):
            index = random.sample(choice,1)
            distance_list = []
            p_center = index[0]
            for n in range(opt.num_points):
                distance_list.append(distance_squre(real_point[m,0,n],p_center))
            distance_order = sorted(enumerate(distance_list), key  = lambda x:x[1])
            
            crop_num_list = []
            for num in range(opt.num_points-opt.crop_point_num):
                crop_num_list.append(distance_order[num+opt.crop_point_num][0])
            indices = torch.LongTensor(crop_num_list)
            input_cropped[m,0]=torch.index_select(real_point[m,0],0,indices)
            
        real_point = torch.squeeze(real_point,1)
        input_cropped = torch.squeeze(input_cropped,1)
        real_point = Variable(real_point,requires_grad=True)  
        input_cropped = Variable(input_cropped,requires_grad=True) 
        real_point = real_point.to(device)
        input_cropped = input_cropped.to(device)        
        optimizer.zero_grad()
        reconstruction=LGAN_autoencoder(input_cropped)        
        errG = criterion_PointLoss(reconstruction,real_point)
        errG.backward()
        optimizer.step()
        print('[%d/%d][%d/%d]  Loss_G: %.4f '
              % (epoch, opt.n_epochs, batch_id, len(dataloader), errG))  
        f=open('LGAN_loss.txt','a')
        f.write('\n'+'[%d/%d][%d/%d]  Loss_G: %.4f '
              % (epoch, opt.n_epochs, batch_id, len(dataloader), errG))
        
        if batch_id % 20 ==0:
            LGAN_autoencoder.eval()
            print('After, ',batch_id,'-th batch')
            f.write('\n'+'After, '+str(batch_id)+'-th batch')
            for i, data in enumerate(test_dataloader, 0):
                real_point, target = data
                real_point = torch.unsqueeze(real_point,1)
                p_origin = [0,0,0]
                batch_size = real_point.size()[0]
                input_cropped =torch.FloatTensor(batch_size, 1, opt.num_points-opt.crop_point_num, 3)
                choice =[torch.Tensor([1,0,0]),torch.Tensor([0,0,1]),torch.Tensor([1,0,1]),torch.Tensor([-1,0,0]),torch.Tensor([-1,1,0])]
                for m in range(batch_size):
                    index = random.sample(choice,1)
                    distance_list = []
                    p_center = index[0]
                    for n in range(opt.num_points):
                        distance_list.append(distance_squre(real_point[m,0,n],p_center))
                    distance_order = sorted(enumerate(distance_list), key  = lambda x:x[1])
                    
                    crop_num_list = []
                    for num in range(opt.num_points-opt.crop_point_num):
                        crop_num_list.append(distance_order[num+opt.crop_point_num][0])
                    indices = torch.LongTensor(crop_num_list)
                    input_cropped[m,0]=torch.index_select(real_point[m,0],0,indices)
                    
                real_point = torch.squeeze(real_point,1)
                input_cropped = torch.squeeze(input_cropped,1)
        
                real_point = real_point.to(device)
                input_cropped = input_cropped.to(device)
                reconstruction=LGAN_autoencoder(input_cropped) 
                errG = criterion_PointLoss(reconstruction,real_point)  
                print('test result:',errG)
                f.write('\n'+'test result:  %.4f'%(errG))
                break
        f.close()
    if epoch% 5 == 0:
        torch.save({'epoch':epoch+1,
                'state_dict':LGAN_autoencoder.module.state_dict()},
                'Trained_Recon_Model_LGAN/LGAN_ae'+str(epoch)+'.pth' )                
                    
                    
                    
