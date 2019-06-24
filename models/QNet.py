import os, sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class QNet(nn.Module):
    def __init__(self, input_dim, hidden_dims=[32, 32], num_actions=4):
        super(QNet, self).__init__()
        self.num_actions = num_actions
        self.l1 = nn.Linear(input_dim, hidden_dims[0])
        self.l2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.l3 = nn.Linear(hidden_dims[1], num_actions)

        if torch.cuda.is_available():
            self.dtype = torch.cuda.FloatTensor
        else:
            self.dtype = torch.FloatTensor

    def forward(self, x):
        t = torch.relu(self.l1(x))
        t = torch.relu(self.l2(t))
        t = self.l3(t)
        return t

class QNetConv(nn.Module):
    def __init__(self, input_dim, hidden_dims=[32, 32], num_actions=4):
        super(QNetConv, self).__init__()
        self.num_actions = num_actions
        self.conv1 = nn.Conv2d(input_dim, hidden_dims[0], 3, padding=1, stride=2)
        self.conv2 = nn.Conv2d(hidden_dims[0], hidden_dims[1], 3, padding=1, stride=2)
        #self.conv2 = nn.Conv2d(hidden_dims[0], hidden_dims[1], 3, padding=1)
        #self.conv1 = nn.Conv2D(hidden_dims[1], hidden_dims[2])
        self.l1 = nn.Linear(hidden_dims[1]*7*7, num_actions)
        #self.l1 = nn.Linear(hidden_dims[1]*6*6, num_actions)

        if torch.cuda.is_available():
            self.dtype = torch.cuda.FloatTensor
        else:
            self.dtype = torch.FloatTensor

    def forward(self, x):
        t = torch.relu(self.conv1(x))
        t = torch.relu(self.conv2(t))
        t = t.view(x.size(0), -1)
        t = self.l1(t)
        return t

class QNetConvWithID(nn.Module):
    def __init__(self, input_dim, hidden_dims=[32, 32], num_actions=4, agent_emb_dim=5, agent_emb_hidden=16):
        super(QNetConvWithID, self).__init__()
        self.num_actions = num_actions
        self.conv1 = nn.Conv2d(input_dim, hidden_dims[0], 3, padding=1, stride=2)
        self.conv2 = nn.Conv2d(hidden_dims[0], hidden_dims[1], 3, padding=1, stride=2)
        #self.conv2 = nn.Conv2d(hidden_dims[0], hidden_dims[1], 3, padding=1)
        #self.conv1 = nn.Conv2D(hidden_dims[1], hidden_dims[2])
        self.embedding = nn.Linear(agent_emb_dim, agent_emb_hidden)
        self.l1 = nn.Linear(hidden_dims[1]*7*7+agent_emb_hidden, num_actions)
        #self.l1 = nn.Linear(hidden_dims[1]*6*6, num_actions)

        if torch.cuda.is_available():
            self.dtype = torch.cuda.FloatTensor
        else:
            self.dtype = torch.FloatTensor

    def forward(self, x, id_):
        t = torch.relu(self.conv1(x))
        t = torch.relu(self.conv2(t))
        t = t.view(x.size(0), -1)
        emb = torch.relu(self.embedding(id_))
        t = torch.cat([t, emb], 1)
        t = self.l1(t)
        return t

