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
parser.add_argument('--model', type=str, default='Trained_Recon_Model_Capsule/3dCapsule120.pth', help='model path')
parser.add_argument('--dataset', type=str, default='shapenet_part', help='dataset: shapenet_part, shapenet_core13, shapenet_core55')
opt = parser.parse_args()
print(opt)



test_dset = shapenet_part_loader.PartDataset( root='../dataset/shapenetcore_partanno_segmentation_benchmark_v0/',classification=True, class_choice='Motorbike', npoints=opt.num_points, split='test')
test_dataloader = torch.utils.data.DataLoader(test_dset, batch_size=opt.batch_size,
                                         shuffle=True,num_workers = int(opt.workers))


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
capsule_net = PointCapsNet(opt.prim_caps_size, opt.prim_vec_size, opt.latent_caps_size, opt.latent_caps_size, opt.num_points)
capsule_net.load_state_dict(torch.load(opt.model,map_location=lambda storage, location: storage)['state_dict'])      
print("Let's use", torch.cuda.device_count(), "GPUs!")
capsule_net.to(device)
capsule_net = torch.nn.DataParallel(capsule_net)
capsule_net.eval()


criterion_PointLoss = PointLoss().to(device)
errG_min = 100
n = 0
for i, data in enumerate(test_dataloader, 0):
        
    real_point, target = data
    real_point = torch.unsqueeze(real_point, 1)
    input_cropped =torch.FloatTensor(opt.batch_size, 1, opt.num_points-opt.crop_point_num, 3)
    input_cropped_ours =torch.FloatTensor(opt.batch_size, 1, opt.num_points, 3)
    input_cropped_ours.resize_(real_point.size()).copy_(real_point)
    batch_size = real_point.size()[0]
    p_origin = [0,0,0]
    choice =[torch.Tensor([1,0,0]),torch.Tensor([0,0,1]),torch.Tensor([1,0,1])]   
#    points = [x for x in range(0,opt.pnum-1)]
#    choice =random.sample(points,5)
    index = random.sample(choice,1)
    distance_list = []
#    p_center  = real_point[0,0,index]
    p_center = index[0]
    for num in range(opt.num_points):
        distance_list.append(distance_squre(real_point[0,0,num],p_center))
    distance_order = sorted(enumerate(distance_list), key = lambda x:x[1])
    
    for sp in range(opt.crop_point_num):
        input_cropped_ours.data[0,0,distance_order[sp][0]] = torch.FloatTensor([0,0,0])
    
    crop_num_list = []
    for num in range(opt.num_points-opt.crop_point_num):
        crop_num_list.append(distance_order[num+opt.crop_point_num][0])
    indices = torch.LongTensor(crop_num_list)
    input_cropped[0,0]=torch.index_select(real_point[0,0],0,indices)
    
    
    real_point = torch.squeeze(real_point,1)
    input_cropped = torch.squeeze(input_cropped,1)
    input_cropped = input_cropped.transpose(2, 1)
    real_point = real_point.to(device)
    input_cropped = input_cropped.to(device)     
    codewords,reconstruction=capsule_net(input_cropped) 
    reconstruction_ = reconstruction.transpose(2, 1).contiguous()
    reconstruction_ = reconstruction_.cuda()
    input_cropped = input_cropped.transpose(2, 1)
    real_point  = real_point.cuda()
#    real_center_key =real_center_key.cuda()
    errG = criterion_PointLoss(reconstruction_,real_point)
    errG = errG.cpu()
    if errG.detach().numpy()>errG_min:
        pass
    
    else:
        errG_min = errG.detach().numpy()
        print(errG_min)
        reconstruction_ =reconstruction_.cpu()
        np_recon = reconstruction_.data[0].detach().numpy() 
        input_cropped = input_cropped.cpu()
        np_crop = input_cropped.data[0].detach().numpy() 
#            num_real = np.array([[len(np_real)]])
        real_point = real_point.cpu()
        np_real = real_point.data[0].detach().numpy()
        
        input_cropped_ours = torch.squeeze(input_cropped_ours,1)
        input_cropped_ours = input_cropped_ours.cpu()
        np_crop_ours = input_cropped_ours[0].detach().numpy()

        n=n+1
        k = 0
#            num_crop = np.array([[len(np_crop)]])
        #np_ini = real_point[0,0].detach().numpy() #1024
        #np_crop = np.
        np.savetxt('test_example_capsule/crop'+str(n)+'.csv', np_crop, fmt = "%f,%f,%f")
        np.savetxt('test_example_capsule/recon'+str(n)+'.csv', np_recon, fmt = "%f,%f,%f")
        np.savetxt('test_example_capsule/real'+str(n)+'.csv', np_real, fmt = "%f,%f,%f")
        np.savetxt('test_example_capsule/crop-ours'+str(n)+'.csv', np_crop_ours, fmt = "%f,%f,%f")
    
    
    
    