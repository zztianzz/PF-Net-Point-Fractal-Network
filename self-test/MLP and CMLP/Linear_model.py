 #!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data



class Linear_autoencoder(nn.Module):
    def __init__(self,num_inputs,num_outputs):
        super(Linear_autoencoder,self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.conv1 = torch.nn.Conv2d(1, 64, (1, 3))
        self.conv2 = torch.nn.Conv2d(64, 128, 1)
        self.conv3 = torch.nn.Conv2d(128, 256, 1)
        self.conv4 = torch.nn.Conv2d(256, 512, 1)
        self.conv5 = torch.nn.Conv2d(512, 1024, 1)
        self.maxpool1 = torch.nn.MaxPool2d((self.num_inputs, 1), 1)
        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, self.num_outputs*3)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)
        self.bn5 = nn.BatchNorm2d(1024)
        self.bn_1 = nn.BatchNorm1d(1024)
        
    def forward(self,x):
        x = torch.unsqueeze(x,1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.maxpool1(x)
        x = torch.squeeze(x,2)
        x = torch.squeeze(x,2)
#        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn_1(self.fc1(x)))
        x = self.fc2(x)
        x = x.reshape(-1,self.num_outputs,3)     
        return x

class CMLP_autoencoder(nn.Module):
    def __init__(self,num_inputs,num_outputs):
        super(CMLP_autoencoder,self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.conv1 = torch.nn.Conv2d(1, 64, (1, 3))
        self.conv2 = torch.nn.Conv2d(64, 128, 1)
        self.conv3 = torch.nn.Conv2d(128, 256, 1)
        self.conv4 = torch.nn.Conv2d(256, 512, 1)
        self.conv5 = torch.nn.Conv2d(512, 1024, 1)
        self.maxpool1 = torch.nn.MaxPool2d((self.num_inputs, 1), 1)
        self.fc1 = nn.Linear(1920, 1024)
        self.fc2 = nn.Linear(1024, self.num_outputs*3)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)
        self.bn5 = nn.BatchNorm2d(1024)
        self.bn_1 = nn.BatchNorm1d(1024)
        
    def forward(self,x):
        x = torch.unsqueeze(x,1)
        x = F.relu(self.bn1(self.conv1(x)))
        x_128 = F.relu(self.bn2(self.conv2(x)))
        x_256 = F.relu(self.bn3(self.conv3(x_128)))
        x_512 = F.relu(self.bn4(self.conv4(x_256)))
        x_1024 = F.relu(self.bn5(self.conv5(x_512)))
        x_128 = torch.squeeze(self.maxpool1(x_128),2)
        x_256 = torch.squeeze(self.maxpool1(x_256),2)
        x_512 = torch.squeeze(self.maxpool1(x_512),2)
        x_1024 = torch.squeeze(self.maxpool1(x_1024),2)
        L = [x_1024,x_512,x_256,x_128]
        x = torch.cat(L,1)
        x = torch.squeeze(x,2)
#        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn_1(self.fc1(x)))
        x = self.fc2(x)
        x = x.reshape(-1,self.num_outputs,3)     
        return x
    
if __name__=='__main__':
    a  = torch.randn(2,1536,3)
    Linear = CMLP_autoencoder(1536,512)
    b = Linear(a)
    

