# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 21:26:49 2019

@author: brand
"""
import os
import PIL
import numpy as np
import torch

traindata = 'D:\DisasterTorch\Train'
validdata = 'D:\DisasterTorch\Valid'
trainK = traindata + '\\K'
trainR = traindata + '\\R'
validK = validdata + '\\K'
validR = validdata + '\\R'
classes = {'K':0, 'R':1}


def clean_data():
    """
    There are odd cases in the data where a few files do not share the same
    format as the rest, this function cleans the data by getting rid of these
    files
    """
    
    for path in [trainK, trainR]:
        for file in os.listdir(path):
            im = PIL.Image.open(path + '\\' + file)
            im = im.convert("RGB")
            array = np.array(im, dtype='float32')
            if array.shape[0] != 416 or array.shape[1] != 549:
                os.remove(path + '\\' + file)

def get_data(mode, batch_size, currbatch):
    assert mode in ('train', 'valid')
    assert batch_size % 2 == 0
    chunk = int(batch_size/2)
    if mode == 'train':
        paths = [trainK, trainR]
        inputs = list()
        labels = list()
        l = 0
        start = currbatch * chunk
        end = start+chunk
        for path in paths:
                for file in os.listdir(path)[start:end]: 
                    im = PIL.Image.open(path + '\\' + file)
                    im = im.convert("RGB")
                    array = np.array(im, dtype='float32')
                    inputs.append(array)
                    if l == 0:
                        labels.append(0)
                    elif l == 1:
                        labels.append(1)
                
                l += 1
    inputs = torch.Tensor(inputs)
    inputs = inputs.view(batch_size, -1, 416, 549)
    
    return inputs, torch.Tensor(labels).long()        
        
if __name__ == '__main__':
    ins, ls = get_data('train', 10, 1)
    print(ins, ls)
