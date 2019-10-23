## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        #considering input size of 224*224 image
        
        self.conv1 = nn.Conv2d(1, 32, 5)        #convolution layer with 1 in-feature and 32 out-features 
      

        self.conv2 = nn.Conv2d(32, 64,5)        #convolution layer with 32 in-feature and 64 out-features
        

        self.conv3 = nn.Conv2d(64, 128,5)       #convolution layer with 64 in-feature and 128 out-features 
        
        self.pool  = nn.MaxPool2d(2,2)          #maxpooling layer
        
        self.dropout = nn.Dropout(p = 0.2)      #dropout layer
        
        self.fc1 = nn.Linear(73728,4096)        #linear layer with input of 73728 features and output of 4096 
        
        
        self.fc2 = nn.Linear(4096,1024)         #linear layer with input of 4096 features and output of 1024
         
        
        self.fc3 = nn.Linear(1024,136)         #linear layer with input of 1024 features and output of 136
      
        
        

        
    def forward(self, x):
        
        
        x = self.pool(F.relu(self.conv1(x)))   #conv1 with relu activation  and maxpooling
        x = self.pool(F.relu(self.conv2(x)))   #conv2 with relu activation  and maxpooling
        x = self.dropout(x)                    #conv2 with dropout layer
        x = self.pool(F.relu(self.conv3(x)))   #conv3 with relu activation  and maxpooling
        x = self.dropout(x)                    #conv3 with dropout layer
        
        x = x.view(x.size(0),-1)               #flattening the image to feed into linear layer
        
        x = F.relu(self.fc1(x))                #fc1 with relu activation
        x = self.dropout(x)                    #fc1 with dropout layer
        x = F.relu(self.fc2(x))                #fc2 with relu activation 
        x = self.dropout(x)                    #fc2 with dropout layer
        x = self.fc3(x)                        #final dense linear layer
        
        
        return x
