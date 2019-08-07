# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 18:20:23 2019

@author: brand
"""

import torch
import numpy as np
import torch.nn.functional as F
import DataFormat as DF

seed = 37
np.random.seed(seed)
torch.manual_seed(seed)

traindata = 'D:\DisasterTorch\Train'
validdata = 'D:\DisasterTorch\Valid'


trainK = traindata + '\\K'
trainR = traindata + '\\R'
validK = validdata + '\\K'
validR = validdata + '\\R'

BATCH_SIZE = 10

classes = ('K', 'R')

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        self.conv1 = torch.nn.Conv2d(3, 18, kernel_size=3, stride=1, padding=1)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = torch.nn.Linear(18 * 208 * 274, 64)
        self.fc2 = torch.nn.Linear(64, 2)
        
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = x.view(-1, 18 * 208 *274)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x


def train_model(model, max_epoch):
    i = 0 
    while i < 1:
        save = input('Save model? [Y/N]: ')
        if save in ('Y', 'y', 'N', 'n'):
            i += 1
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    objective_function = torch.nn.CrossEntropyLoss()
    print('Beginning Training...')
    for i in range(max_epoch):    
        minibatches = int(800 / BATCH_SIZE) * 2
        for j in range(minibatches):
            data = DF.get_data('train', BATCH_SIZE, j)  
            inputs = data[0].cuda()
            labels = data[1].cuda()         
            model.zero_grad()
            output = model(inputs)    
            loss = objective_function(output, labels) 
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
        print(loss)
    if save in ('Y', 'y'):
        torch.save(model.state_dict(),'./SavedModels/ModelParams.py' )
    elif save in ('N', 'n'):
        pass

if __name__ == "__main__":

    model = CNN()
    train_model(model.cuda(), 1)
    