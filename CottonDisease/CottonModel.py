import torch
import torch.nn as nn
import torch.nn.functional as F

class CottonTypeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,16,3,1)
        self.conv2 = nn.Conv2d(16,32,3,1)
        self.conv3 = nn.Conv2d(32,64,3,1)
        self.conv4 = nn.Conv2d(64,64,3,1)
        self.conv5 = nn.Conv2d(64,64,3,1)
        self.fc1 = nn.Linear(1600,128)
        self.fc2 = nn.Linear(128,4)
    
    def forward(self,X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X,2,2)
        
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X,2,2)
        
        X = F.relu(self.conv3(X))
        X = F.max_pool2d(X,2,2)
        
        X = F.relu(self.conv4(X))
        X = F.max_pool2d(X,2,2)
        
        X = F.relu(self.conv5(X))
        X = F.max_pool2d(X,2,2)
        
        X = X.view(-1,1600)
        
        X = F.relu(self.fc1(X))
        X = self.fc2(X)
        return F.log_softmax(X,dim=1)