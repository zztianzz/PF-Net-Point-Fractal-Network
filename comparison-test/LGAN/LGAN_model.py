 #!/usr/bin/env python3
# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data



class LGAN_autoencoder(nn.Module):
    def __init__(self,num_inputs,num_outputs):
        super(LGAN_autoencoder,self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.conv1 = torch.nn.Conv2d(1, 64, (1, 3))
        self.conv2 = torch.nn.Conv2d(64, 128, 1)
        self.conv3 = torch.nn.Conv2d(128, 1024, 1)
        self.maxpool1 = torch.nn.MaxPool2d((self.num_inputs, 1), 1)
        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, self.num_outputs*3)
#        self.bn1 = nn.BatchNorm2d(64)
#        self.bn2 = nn.BatchNorm2d(128)
#        self.bn3 = nn.BatchNorm2d(1024)
#        self.bn4 = nn.BatchNorm1d(1024)
        
    def forward(self,x):
        x = torch.unsqueeze(x,1)
#        x = F.relu(self.bn1(self.conv1(x)))
#        x = F.relu(self.bn2(self.conv2(x)))
#        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.maxpool1(x)
        x = torch.squeeze(x,2)
        x = torch.squeeze(x,2)
#        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.reshape(-1,self.num_outputs,3)     
        return x
    
if __name__=='__main__':
    a  = torch.randn(64,1536,3)
    LGAN = LGAN_autoencoder(1536,2048)
    b = LGAN(a)
    

