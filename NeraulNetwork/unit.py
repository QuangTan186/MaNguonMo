import numpy as np
import torch
from torch import Tensor
import torch.nn as nn

def initializeWeight(size):
    w = np.random.standard_normal(size=size) *0.1
    return torch.from_numpy(w).float()
class dense(nn.Linear):
    def __init__(self,input,output):
        super(dense, self).__init__(input,output)
        self.w = initializeWeight((input,output))
        self.__class__ = nn.Linear
    def forward(self,x):
        b = torch.zeros((x.shape[0],1))
        return  (x.mm(self.w) + b)