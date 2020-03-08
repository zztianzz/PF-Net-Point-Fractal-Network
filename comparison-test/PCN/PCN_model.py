 #!/usr/bin/env python3
# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data



class Encoder(nn.Module):
    def __init__(self,num_inputs):
        super(Encoder,self).__init__()
        self.num_inputs = num_inputs
        self.conv1 = torch.nn.Conv2d(1, 128, (1, 3))
        self.conv2 = torch.nn.Conv2d(128, 256, 1)
        self.maxpool1 = torch.nn.MaxPool2d((self.num_inputs, 1), 1)
        self.conv3 = torch.nn.Conv2d(512, 512, 1)
        self.conv4 = torch.nn.Conv2d(512, 1024, 1)
        self.maxpool2 = torch.nn.MaxPool2d((self.num_inputs, 1), 1)
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(512)
        self.bn4 = nn.BatchNorm2d(1024)
    def forward(self,x):
        x = torch.unsqueeze(x,1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x_256 = self.maxpool1(x)
        feature_global = x_256.repeat(1,1,self.num_inputs,1)
        feature = torch.cat((x,feature_global),1)
        feature = F.relu(self.bn3(self.conv3(feature)))
        feature = F.relu(self.bn4(self.conv4(feature)))
        feature = self.maxpool2(feature)
        feature = torch.squeeze(feature,2)
        feature = torch.squeeze(feature,2)
        return feature
    

class Autoencoder(nn.Module):
    def __init__(self,num_inputs,num_coarses,num_fines,grid_size):
        super(Autoencoder,self).__init__()
        self.num_inputs = num_inputs
        self.num_coarses = num_coarses
        self.num_fines = num_fines
        self.grid_size = 4
        self.Encoder = Encoder(num_inputs)
        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, self.num_coarses*3)
        self.conv1 = torch.nn.Conv1d(1029,512,1)
        self.conv2 = torch.nn.Conv1d(512,3,1)   
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)
       
      

    def forward(self,x):
        device = x.device
        feature = self.Encoder(x)
    
        coarses = F.relu(self.bn1(self.fc1(feature)))
        coarses = self.fc2(coarses)
        coarses = coarses.reshape(-1,self.num_coarses,3)
        
        batch_size = x.size()[0]
        grid =  torch.meshgrid([torch.linspace(-0.05,0.05,self.grid_size), torch.linspace(-0.05,0.05,self.grid_size)])
        grid_ = torch.cat((grid[1].reshape(16,1),grid[0].reshape(16,1)),1)
        grid_ = torch.unsqueeze(grid_,0)
        num1 = int(self.num_fines/self.grid_size**2)        
        grid_feature = grid_.repeat(batch_size,num1,1)
#        grid_feature.to(device)
        
        point_feat = torch.unsqueeze(coarses,2)
        num2 = int(self.num_fines/self.num_coarses)
        point_feat = point_feat.repeat(1,1,num2,1)
        point_feat = point_feat.reshape(-1,self.num_fines,3)
        
        global_feat = torch.unsqueeze(feature,1)
        global_feat = global_feat.repeat(1,self.num_fines,1)
        
        grid_feature =grid_feature.to(device)
        point_feat = point_feat.to(device)
        global_feat = global_feat.to(device)
        feat = torch.cat([grid_feature,point_feat,global_feat],2)
        feat = feat.transpose(1,2).contiguous()
        
        center = torch.unsqueeze(coarses,2)
        num3 = int(self.num_fines/self.num_coarses)
        center = center.repeat(1,1,num3,1)
        center = center.reshape(-1,self.num_fines,3)
        
        fine = F.relu(self.bn2(self.conv1(feat)))
        fine = self.conv2(fine)
        
        fine = fine.transpose(1,2).contiguous()
        fine = fine+center
        
        return coarses, fine
