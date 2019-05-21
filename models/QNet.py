import os, sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class QNet(nn.Module):
    def __init__(self, input_dim, hidden_dims=[32, 32], action_size=4):
        super(QNet, self).__init__()
        self.num_actions = action_size
        self.l1 = nn.Linear(input_dim, hidden_dims[0])
        self.l2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.l3 = nn.Linear(hidden_dims[1], action_size)

        if torch.cuda.is_available():
            self.dtype = torch.cuda.FloatTensor
        else:
            self.dtype = torch.FloatTensor

    def forward(self, x):
        t = torch.sigmoid(self.l1(x))
        t = torch.sigmoid(self.l2(t))
        t = self.l3(t)
        return t

