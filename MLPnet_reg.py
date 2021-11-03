
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 18:11:28 2019

@author: jkf
"""
import torch.nn as nn

class NET(nn.Module):
    def __init__(self, input_size, hidden_size, block, output_size):
        super(NET, self).__init__()
        # self.bn0 = nn.BatchNorm1d(input_size)
        # self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc0 = nn.Linear(input_size, hidden_size,bias=True) 
        self.fc1_1 = nn.Linear(hidden_size, hidden_size,bias=True) 
        self.fc1_2 = nn.Linear(hidden_size, hidden_size,bias=True) 
        self.fc2 = nn.Linear(hidden_size, output_size,bias=True)  
#        self.activ = nn.ReLU()
        self.activ = nn.Tanh()
        self.dropout = nn.Dropout(0.4)
        self.block = block
        
    def forward(self, out):

        out = self.fc0(out)
        out = self.activ(out)
        for i in range(self.block):
            res=out
            out = self.fc1_1(out)
            out = self.activ(out)
            out = self.fc1_2(out)
            out+=res
            out = self.activ(out)

        out = self.fc2(out)
        return out
