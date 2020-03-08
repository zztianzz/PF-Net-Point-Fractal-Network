#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from __future__ import print_function
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
from collections import OrderedDict
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append('..')

from utils import PointLoss


#from modules.nnd import NNDModule
#distChamfer = NNDModule()
USE_CUDA = True


class ConvLayer(nn.Module):
    def __init__(self):
        super(ConvLayer, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x


class PrimaryPointCapsLayer(nn.Module):
    def __init__(self, prim_vec_size=8, num_points=2048):
        super(PrimaryPointCapsLayer, self).__init__()
        self.capsules = nn.ModuleList([
            torch.nn.Sequential(OrderedDict([
                ('conv3', torch.nn.Conv1d(128, 1024, 1)),
                ('bn3', nn.BatchNorm1d(1024)),
                ('mp1', torch.nn.MaxPool1d(num_points)),
            ]))
            for _ in range(prim_vec_size)])
    
    def forward(self, x):
        u = [capsule(x) for capsule in self.capsules]
        u = torch.stack(u, dim=2)
        return self.squash(u.squeeze())
    
    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm * input_tensor / \
            ((1. + squared_norm) * torch.sqrt(squared_norm))
        if(output_tensor.dim() == 2):
            output_tensor = torch.unsqueeze(output_tensor, 0)
        return output_tensor


class LatentCapsLayer(nn.Module):
    def __init__(self, latent_caps_size=16, prim_caps_size=1024, prim_vec_size=16, latent_vec_size=64):
        super(LatentCapsLayer, self).__init__()
        self.prim_vec_size = prim_vec_size
        self.prim_caps_size = prim_caps_size
        self.latent_caps_size = latent_caps_size
        self.W = nn.Parameter(0.01*torch.randn(latent_caps_size, prim_caps_size, latent_vec_size, prim_vec_size))

    def forward(self, x):
        u_hat = torch.squeeze(torch.matmul(self.W, x[:, None, :, :, None]), dim=-1)
        u_hat_detached = u_hat.detach()
        b_ij = Variable(torch.zeros(x.size(0), self.latent_caps_size, self.prim_caps_size)).cuda()
        num_iterations = 3
        for iteration in range(num_iterations):
            c_ij = F.softmax(b_ij, 1)
            if iteration == num_iterations - 1:
                v_j = self.squash(torch.sum(c_ij[:, :, :, None] * u_hat, dim=-2, keepdim=True))
            else:
                v_j = self.squash(torch.sum(c_ij[:, :, :, None] * u_hat_detached, dim=-2, keepdim=True))
                b_ij = b_ij + torch.sum(v_j * u_hat_detached, dim=-1)
        return v_j.squeeze(-2)
    
    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm * input_tensor / \
            ((1. + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor


class PointGenCon(nn.Module):
    def __init__(self, bottleneck_size=2500):
        self.bottleneck_size = bottleneck_size
        super(PointGenCon, self).__init__()
        self.conv1 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size, 1)
        self.conv2 = torch.nn.Conv1d(self.bottleneck_size, int(self.bottleneck_size/2), 1)
        self.conv3 = torch.nn.Conv1d(int(self.bottleneck_size/2), int(self.bottleneck_size/4), 1)
        self.conv4 = torch.nn.Conv1d(int(self.bottleneck_size/4), 3, 1)
        self.th = torch.nn.Tanh()
        self.bn1 = torch.nn.BatchNorm1d(self.bottleneck_size)
        self.bn2 = torch.nn.BatchNorm1d(int(self.bottleneck_size/2))
        self.bn3 = torch.nn.BatchNorm1d(int(self.bottleneck_size/4))

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.th(self.conv4(x))
        return x

class CapsDecoder(nn.Module):
    def __init__(self, latent_caps_size, latent_vec_size, num_points):
        super(CapsDecoder, self).__init__()
        self.latent_caps_size = latent_caps_size
        self.bottleneck_size=latent_vec_size
        self.num_points = num_points
        self.nb_primitives=int(num_points/latent_caps_size)
        self.decoder = nn.ModuleList(
            [PointGenCon(bottleneck_size=self.bottleneck_size+2) for i in range(0, self.nb_primitives)])
    def forward(self, x):
        outs = []
        for i in range(0, self.nb_primitives):
            rand_grid = Variable(torch.cuda.FloatTensor(x.size(0), 2, self.latent_caps_size))
            rand_grid.data.uniform_(0, 1)
            y = torch.cat((rand_grid, x.transpose(2, 1)), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs, 2).contiguous()

    
class PointCapsNet(nn.Module):
    def __init__(self, prim_caps_size, prim_vec_size, latent_caps_size, latent_vec_size, num_points):
        super(PointCapsNet, self).__init__()
        self.conv_layer = ConvLayer()
        self.primary_point_caps_layer = PrimaryPointCapsLayer(prim_vec_size, num_points)
        self.latent_caps_layer = LatentCapsLayer(latent_caps_size, prim_caps_size, prim_vec_size, latent_vec_size)
        self.caps_decoder = CapsDecoder(latent_caps_size,latent_vec_size, num_points)

    def forward(self, data):
        x1 = self.conv_layer(data)
        x2 = self.primary_point_caps_layer(x1)
        latent_capsules = self.latent_caps_layer(x2)
        reconstructions = self.caps_decoder(latent_capsules)
        return latent_capsules, reconstructions

    def loss(self, data, reconstructions):
         return self.reconstruction_loss(data, reconstructions)

    def reconstruction_loss(self, data, reconstructions):
        data_ = data.transpose(2, 1).contiguous()
        reconstructions_ = reconstructions.transpose(2, 1).contiguous()
        dist1, dist2 = distChamfer(data_, reconstructions_)
        loss = (torch.mean(dist1)) + (torch.mean(dist2))
        return loss 
    
# This is a single network which can decode the point cloud from pre-saved latent capsules
class PointCapsNetDecoder(nn.Module):
    def __init__(self, prim_caps_size, prim_vec_size, digit_caps_size, digit_vec_size, num_points):
        super(PointCapsNetDecoder, self).__init__()
        self.caps_decoder = CapsDecoder(digit_caps_size,digit_vec_size, num_points)
    def forward(self, latent_capsules):
        reconstructions = self.caps_decoder(latent_capsules)
        return  reconstructions

if __name__ == '__main__':
    USE_CUDA = True
    p = PointLoss()
    batch_size=8
    
    prim_caps_size=1024
    prim_vec_size=16
    
    latent_caps_size=32
    latent_vec_size=16
    
    num_points=2048

    point_caps_ae = PointCapsNet(prim_caps_size,prim_vec_size,latent_caps_size,latent_vec_size,num_points)
    point_caps_ae=torch.nn.DataParallel(point_caps_ae).cuda()
    
    rand_data=torch.rand(batch_size,num_points, 3) 
    rand_data = Variable(rand_data)
    rand_data = rand_data.transpose(2, 1)
    rand_data=rand_data.cuda()
    
    codewords,reconstruction=point_caps_ae(rand_data)
   
    rand_data_ = rand_data.transpose(2, 1).contiguous()
    reconstruction_ = reconstruction.transpose(2, 1).contiguous()
    
    print(p(rand_data_,reconstruction_))
    
#    dist1, dist2 = distChamfer(rand_data_, reconstruction_)
#    loss = (torch.mean(dist1)) + (torch.mean(dist2))
#    print(loss.item())