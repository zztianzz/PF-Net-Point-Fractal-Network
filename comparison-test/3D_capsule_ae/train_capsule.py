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
from pointcapsnet_ae import PointCapsNet
import data_utils as d_utils

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=7, help='input batch size')
parser.add_argument('--n_epochs', type=int, default=300, help='number of epochs to train for')
parser.add_argument('--workers', type=int,default=2, help='number of data loading workers')
parser.add_argument('--prim_caps_size', type=int, default=1024, help='number of primary point caps')
parser.add_argument('--prim_vec_size', type=int, default=16, help='scale of primary point caps')
parser.add_argument('--latent_caps_size', type=int, default=64, help='number of latent caps')
parser.add_argument('--latent_vec_size', type=int, default=64, help='scale of latent caps')

parser.add_argument('--num_points', type=int, default=2048, help='input point set size')
parser.add_argument('--crop_point_num',type=int,default=512,help='number of crop points ')
parser.add_argument('--outf', type=str, default='tmp_checkpoints', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--dataset', type=str, default='shapenet_part', help='dataset: shapenet_part, shapenet_core13, shapenet_core55')
opt = parser.parse_args()

resume_epoch=0
USE_CUDA = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
capsule_net = PointCapsNet(opt.prim_caps_size, opt.prim_vec_size, opt.latent_caps_size, opt.latent_caps_size, opt.num_points)

if opt.model != '':
    capsule_net.load_state_dict(torch.load(opt.model,map_location=lambda storage, location: storage)['state_dict'])  
    resume_epoch = torch.load(opt.model)['epoch']
    
if USE_CUDA:       
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    capsule_net.to(device)
    capsule_net = torch.nn.DataParallel(capsule_net)
    



dset = shapenet_part_loader.PartDataset( root='../dataset/shapenetcore_partanno_segmentation_benchmark_v0/',classification=True, class_choice=None, npoints=opt.num_points, split='train')
assert dset
dataloader = torch.utils.data.DataLoader(dset, batch_size=opt.batch_size,
                                         shuffle=True,num_workers = int(opt.workers))


test_dset = shapenet_part_loader.PartDataset( root='../dataset/shapenetcore_partanno_segmentation_benchmark_v0/',classification=True, class_choice=None, npoints=opt.num_points, split='test')
test_dataloader = torch.utils.data.DataLoader(test_dset, batch_size=opt.batch_size,
                                         shuffle=True,num_workers = int(opt.workers))

print(capsule_net)
criterion_PointLoss = PointLoss().to(device)



for epoch in range(resume_epoch,opt.n_epochs): 
    if epoch < 50:
        optimizer = optim.Adam(capsule_net.parameters(), lr=0.0001)
    elif epoch<150:
        optimizer = optim.Adam(capsule_net.parameters(), lr=0.00001)
    else:
        optimizer = optim.Adam(capsule_net.parameters(), lr=0.000001)
        
    for batch_id, data in enumerate(dataloader):  
        capsule_net.train()
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
        input_cropped = input_cropped.transpose(2, 1)
        real_point = real_point.to(device)
        input_cropped = input_cropped.to(device)        
        optimizer.zero_grad()
        codewords,reconstruction=capsule_net(input_cropped) 
        reconstruction_ = reconstruction.transpose(2, 1).contiguous()
        errG = criterion_PointLoss(reconstruction_,real_point)
        errG.backward()
        optimizer.step()
        print('[%d/%d][%d/%d]  Loss_G: %.4f '
              % (epoch, opt.n_epochs, batch_id, len(dataloader), errG))  
        f=open('loss_capsule.txt','a')
        f.write('\n'+'[%d/%d][%d/%d]  Loss_G: %.4f '
              % (epoch, opt.n_epochs, batch_id, len(dataloader), errG))
        
        if batch_id % 10 ==0:
            capsule_net.eval()
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
                input_cropped = input_cropped.transpose(2, 1)
                real_point = real_point.to(device)
                input_cropped = input_cropped.to(device)
                codewords,reconstruction=capsule_net(input_cropped) 
                reconstruction_ = reconstruction.transpose(2, 1).contiguous()
                errG = criterion_PointLoss(reconstruction_,real_point)  
                print('test result:',errG)
                f.write('\n'+'test result:  %.4f'%(errG))
                break
        f.close()
    if epoch% 10 == 0:
        torch.save({'epoch':epoch+1,
                'state_dict':capsule_net.module.state_dict()},
                'Trained_Recon_Model_Capsule/3dCapsule'+str(epoch)+'.pth' )                
                    
                    
                    