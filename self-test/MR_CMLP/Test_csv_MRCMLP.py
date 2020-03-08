#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import numpy as np
import argparse
import os
import sys
import torch
import torch.nn.parallel
import torch.utils.data
import torchvision.transforms as transforms
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append('../..')
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, 'models')))
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, 'dataloaders')))
import utils
from utils import PointLoss
from utils import distance_squre
import data_utils as d_utils
import ModelNet40Loader
import shapenet_part_loader
from model_recon_noFPN import _netG_noFPN


parser = argparse.ArgumentParser()
#parser.add_argument('--dataset',  default='ModelNet40', help='ModelNet10|ModelNet40|ShapeNet')
parser.add_argument('--dataroot',  default='dataset/train', help='path to dataset')
parser.add_argument('--workers', type=int,default=2, help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--pnum', type=int, default=2048, help='the point number of a sample')
parser.add_argument('--crop_point_num',type=int,default=512,help='0 means do not use else use with this weight')
parser.add_argument('--nc', type=int, default=3)
parser.add_argument('--niter', type=int, default=300, help='number of epochs to train for')
parser.add_argument('--weight_decay', type=float, default=0.001)
parser.add_argument('--learning_rate', default=0.0002, type=float, help='learning rate in training')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
parser.add_argument('--cuda', type = bool, default = False, help='enables cuda')
parser.add_argument('--ngpu', type=int, default=2, help='number of GPUs to use')
parser.add_argument('--netG', default='Train_MRCMLP/point_netG100.pth', help="path to netG (to continue training)")
parser.add_argument('--infile',type = str, default = 'Test_one/crop1.csv')
parser.add_argument('--infile_real',type = str, default = 'Test_one/real1.csv')
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--drop',type=float,default=0.2)
parser.add_argument('--num_scales',type=int,default=3,help='number of scales')
parser.add_argument('--point_scales_list',type=list,default=[2048,1024,512],help='number of points in each scales')
parser.add_argument('--each_scales_size',type=int,default=1,help='each scales size')
parser.add_argument('--wtl2',type=float,default=0.9,help='0 means do not use else use with this weight')
parser.add_argument('--cropmethod', default = 'random_center', help = 'random|center|random_center')
opt = parser.parse_args()
print(opt)

def distance_squre1(p1,p2):
    return (p1[0]-p2[0])**2+(p1[1]-p2[1])**2+(p1[2]-p2[2])**2 

transforms = transforms.Compose(
    [
        d_utils.PointcloudToTensor(),
#        d_utils.PointcloudRotate(axis=np.array([1, 0, 0])),
#        d_utils.PointcloudScale(),
#        d_utils.PointcloudTranslate(),
#        d_utils.PointcloudJitter(),
    ]
)

test_dset = shapenet_part_loader.PartDataset( root='../../dataset/shapenetcore_partanno_segmentation_benchmark_v0/',classification=True, class_choice='Car', npoints=opt.pnum, split='test')
test_dataloader = torch.utils.data.DataLoader(test_dset, batch_size=opt.batchSize,
                                         shuffle=True,num_workers = int(opt.workers))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
point_netG = _netG_noFPN(opt.num_scales,opt.each_scales_size,opt.point_scales_list) 
point_netG = torch.nn.DataParallel(point_netG)
point_netG.to(device)
point_netG.load_state_dict(torch.load(opt.netG,map_location=lambda storage, location: storage)['state_dict'])   
point_netG.eval()

input_cropped1 = np.loadtxt(opt.infile,delimiter=',')
input_cropped1 = torch.FloatTensor(input_cropped1)
input_cropped1 = torch.unsqueeze(input_cropped1, 0)
Zeros = torch.zeros(1,512,3)
input_cropped1 = torch.cat((input_cropped1,Zeros),1)


input_cropped2_idx = utils.farthest_point_sample(input_cropped1,1024,RAN = True)
input_cropped2     = utils.index_points(input_cropped1,input_cropped2_idx)
input_cropped3_idx = utils.farthest_point_sample(input_cropped1,512,RAN = False)
input_cropped3     = utils.index_points(input_cropped1,input_cropped3_idx)
input_cropped4_idx = utils.farthest_point_sample(input_cropped1,256,RAN = True)
input_cropped4     = utils.index_points(input_cropped1,input_cropped4_idx)
input_cropped2 = input_cropped2.to(device)
input_cropped3 = input_cropped3.to(device)      
input_cropped  = [input_cropped1,input_cropped2,input_cropped3]

fake=point_netG(input_cropped)
fake = fake.cuda()



input_cropped2 = input_cropped2.cpu()
input_cropped3 = input_cropped3.cpu()
input_cropped4 = input_cropped4.cpu()

np_crop2 = input_cropped2[0].detach().numpy()
np_crop3 = input_cropped3[0].detach().numpy()
np_crop4 = input_cropped4[0].detach().numpy()

real = np.loadtxt(opt.infile_real,delimiter=',')
real = torch.FloatTensor(real)
real = torch.unsqueeze(real,0)





fake =fake.cpu()
np_fake = fake[0].detach().numpy()
input_cropped1 = input_cropped1.cpu()
np_crop = input_cropped1[0].numpy() 

np.savetxt('Test_one/crop_MRCMLP'+'.csv', np_crop, fmt = "%f,%f,%f")
np.savetxt('Test_one/fake_MRCMLP'+'.csv', np_fake, fmt = "%f,%f,%f")
np.savetxt('Test_one/crop_MRCMLP_txt'+'.txt', np_crop, fmt = "%f,%f,%f")
np.savetxt('Test_one/fake_MRCMLP_txt'+'.txt', np_fake, fmt = "%f,%f,%f")