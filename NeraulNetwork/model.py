import torch
import torch.nn as nn
from NeraulNetwork.unit import dense

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.d1 = dense(input_size,hidden_size)
        self.d2 = dense(hidden_size,hidden_size)
        self.d3 = dense(hidden_size,num_classes)
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.d1(x)
        out = self.relu(out)
        out = self.d2(out)
        out = self.relu(out)  
        out = self.d3(out)
        return out
