'''
Copyright@ Qiao-Mu(Albert) Ren. 
All Rights Reserved.
This is the code to build a neural network.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from dataloader import HairNetDataset


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
        self.branch1_fc1 = nn.Linear(512*32*32, 512)
        self.branch1_fc2 = nn.Linear(512, 512)
        self.branch1_fc3 = nn.Linear(512, 32*32*300)
        # Curvature
        self.branch2_fc1 = nn.Linear(512*32*32, 512)
        self.branch2_fc2 = nn.Linear(512, 512)
        self.branch2_fc3 = nn.Linear(512, 32*32*100)
        
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


def train(root_dir):
    print('This is the programme of training.')
    # build model
    print('Initializing Network...')
    net = Net()
    net.cuda()
    loss = MyLoss()
    loss.cuda()
    # set hyperparameter
    EPOCH = 500
    BATCH_SIZE = 32
    LR = 1e-4
    # load data
    print('Setting Dataset and DataLoader...')
    train_data = HairNetDataset(project_dir=root_dir,train_flag=1,noise_flag=1)
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE)
    # set optimizer    
    optimizer = optim.Adam(net.parameters(), lr=LR)
    loss_list=[]
    print('Training...')
    for i in range(EPOCH):
        epoch_loss = 0.0
        # change learning rate when epoch equals 250
        if i == 250:
            for param_group in optimizer.param_groups:
                param_group['lr'] = LR/2.0
        for j, data in enumerate(train_loader, 0):
            img, convdata, visweight = data
            img.cuda()
            convdata.cuda()
            visweight.cuda()
            # img (batch_size, 3, 256, 256)     
            # convdata (batch_size, 100, 4, 32, 32)
            # visweight (batch_size, 100, 32, 32)

            # zero the parameter gradients
            optimizer.zero_grad()
            output = net(img) #img (batch_size, 100, 4, 32, 32)
            loss = loss(output, convdata, visweight)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            if (j+1)%100 == 0:
                print('epoch: ' + str(i+1) + ', ' + str(BATCH_SIZE*(j+1)) + '/' + str(len(train_data)) + ', loss: ' + str(loss.item()))
                if not os.path.exists(project_dir+'/log.txt'):
                    with open(root_dir+'/log.txt', 'w') as f:
                        f.write('epoch: ' + str(i+1) + ', ' + str(BATCH_SIZE*(j+1)) + '/' + str(len(train_data)) + ', loss: ' + str(loss.item()) + '\n')    
                else:
                    with open(root_dir+'/log.txt', 'a') as f:
                        f.write('epoch: ' + str(i+1) + ', ' + str(BATCH_SIZE*(j+1)) + '/' + str(len(train_data)) + ', loss: ' + str(loss.item()) + '\n')        
        if (i+1)%50 == 0:       
            save_path = root_dir + '/weight/' + str(i+1).zfill(6) + '_weight.pt'
            torch.save(net.state_dict(), save_path)
        loss_list.append(epoch_loss)
    print('Finish...')
    plt.plot(loss_list)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(root_dir + 'loss.png')


def test(root_dir, weight_path):
    print('This is the programme of testing.')
    BATCH_SIZE = 32
    # load model
    print('Building Network...')
    net = Net()
    net.cuda()
    pos_error = MyPosEvaluation()
    pos_error.cuda()
    cur_error = MyCurEvaluation()
    cur_error.cuda()
    print('Loading Network...')
    net.load_state_dict(torch.load(weight_path))
    net.eval()
    print('Loading Dataset...')
    test_data = HairNetDataset(project_dir=root_dir,train_flag=0,noise_flag=0)
    test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE)
    # load testing data
    print('Testing...')
    for i, data in enumerate(test_loader, 0):
        img, convdata, visweight = data
        img.cuda()
        convdata.cuda()
        visweight.cuda()
        output = net(img)
        pos = pos_error(output, convdata)
        cur = cur_error(output, convdata)
        print(str(BATCH_SIZE*(i+1)) + '/' + str(len(test_data)) + ', Position loss: ' + str(pos.item()) + ', Curvature loss: ' + str(cur.item()))


