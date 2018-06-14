#pytorch
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import os

class neural_net(nn.Module):
    def __init__(self,state_size,action_size,output_size):
        """Initialize paramters and build network"""
        super(neural_net, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.output_size = output_size

        self.l1 = nn.Linear(self.state_size, 300)
        self.bn1 = nn.BatchNorm1d(300)
        self.rel1 = nn.ReLU()
        self.l2 = nn.Linear(300+1, 600)
        self.rel2 = nn.ReLU()
        self.l3 = nn.Linear(600, self.output_size)
        self.layers = nn.Sequential(
            self.l1,
            self.bn1,
            self.rel1,
            self.l2,
            self.rel2,
            self.l3)

        self.weight_init()

    def forward(self, state_in, action_in):
        """Pass inputs through network"""
        # import pdb; pdb.set_trace()
        out = self.l1(state_in)
        out = self.bn1(out)
        out = self.rel1(out)
        # import pdb; pdb.set_trace()
        out2 = self.l2(torch.cat((out, action_in),1))
        out2 = self.rel2(out2)
        out3 = self.l3(out2)
        return out3

    def weight_init(self):
        """Initalize weights using xavier initialization"""
        # Xavier: mean = 0 and stdev = 1/sqrt(# inputs)
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight)
                nn.init.constant_(layer.bias,0.1)
