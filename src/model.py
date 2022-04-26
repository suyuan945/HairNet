'''
Copyright@ Qiao-Mu(Albert) Ren. 
All Rights Reserved.
This is the code to build a neural network.
'''
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from .dataloader import HairNetDataset

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # encoder
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 2, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 2, 1)
        self.conv4 = nn.Conv2d(128, 256, 3, 2, 1)
        self.conv5 = nn.Conv2d(256, 512, 3, 2, 1)
        # decoder
        self.fc1 = nn.Linear(1*1*512, 4*4*256)
        self.conv6 = nn.Conv2d(256, 512, 3, 1, 1)
        self.conv7 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv8 = nn.Conv2d(512, 512, 3, 1, 1)
        # MLP
        # Position
        self.branch1_fc1 = nn.Linear(512*32*32, 32)
        self.branch1_fc2 = nn.Linear(32, 32)
        self.branch1_fc3 = nn.Linear(32, 32*32*300)
        # Curvature
        self.branch2_fc1 = nn.Linear(512*32*32, 32)
        self.branch2_fc2 = nn.Linear(32, 32)
        self.branch2_fc3 = nn.Linear(32, 32*32*100)
        
    def forward(self, x):
        # encoder
        x = F.relu(self.conv1(x)) # (batch_size, 32, 128, 128)
        x = F.relu(self.conv2(x)) # (batch_size, 64, 64, 64)
        x = F.relu(self.conv3(x)) # (batch_size, 128, 32, 32)
        x = F.relu(self.conv4(x)) # (batch_size, 256, 16, 16)
        x = F.relu(self.conv5(x)) # (batch_size, 512, 8, 8)
        x = F.max_pool2d(x, 8) # (batch_size, 512, 1, 1)
        # decoder
        x = x.view(-1, 1*1*512)
        x = F.relu(self.fc1(x))
        x = x.view(-1, 256, 4, 4)
        x = F.relu(self.conv6(x)) # (batch_size, 512, 4, 4)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners = False) # (batch_size, 512, 8, 8)
        x = F.relu(self.conv7(x)) # (batch_size, 512, 8, 8)
        x = F.interpolate(x, scale_factor=2, mode='bilinear',align_corners = False) # (batch_size, 512, 16, 16)
        x = F.relu(self.conv8(x)) # (batch_size, 512, 16, 16)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners = False) # (batch_size, 512, 32, 32)
        x = x.view(-1, 512*32*32)
        # MLP
        # Position
        branch1_x = F.relu(self.branch1_fc1(x))
        branch1_x = F.relu(self.branch1_fc2(branch1_x))
        branch1_x = F.relu(self.branch1_fc3(branch1_x))
        branch1_x = branch1_x.view(-1, 100, 3, 32, 32)
        # Curvature
        branch2_x = F.relu(self.branch2_fc1(x))
        branch2_x = F.relu(self.branch2_fc2(branch2_x))
        branch2_x = F.relu(self.branch2_fc3(branch2_x))
        branch2_x = branch2_x.view(-1, 100, 1, 32, 32)
        x = [branch1_x, branch2_x]
        return torch.cat(x, 2) # (batch_size, 100, 4, 32, 32)


class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()
    def forward(self, output, convdata, visweight):
        pos_loss = 0.0
        cur_loss = 0.0
        for i in range(0,32):
            for j in range(0,32):
                pos_loss += (visweight[:,:,i,j].reshape(1,-1).mm(torch.pow((convdata[:,:,0:3,i,j]-output[:,:,0:3,i,j]),2).reshape(-1, 3))).sum()
                cur_loss += (visweight[:,:,i,j].reshape(1,-1).mm(torch.pow((convdata[:,:,3,i,j]-output[:,:,3,i,j]),2).reshape(-1, 1))).sum()
        # print(pos_loss/1024.0, cur_loss/1024.0)       
        return pos_loss/1024.0 + cur_loss/1024.0


class MyPosEvaluation(nn.Module):
    def __init__(self):
        super(MyPosEvaluation, self).__init__()
    def forward(self, output, convdata):
        loss = 0.0
        for i in range(0,32):
            for j in range(0,32):
                loss += torch.mean(torch.abs(convdata[:,:,0:3,i,j]-output[:,:,0:3,i,j]))
        return loss/1024.0


class MyCurEvaluation(nn.Module):
    def __init__(self):
        super(MyCurEvaluation, self).__init__()
    def forward(self, output, convdata):
        loss = 0.0
        for i in range(0,32):
            for j in range(0,32):
                loss += torch.mean(torch.abs(convdata[:,:,3,i,j]-output[:,:,3,i,j]))
        return loss/1024.0
