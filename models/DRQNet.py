import os, sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class DRQNet(nn.Module):
    def __init__(self, input_dim, lstm_input, lstm_out, hidden_dims=[32, 64, 128], num_actions=4, agent_emb_dim=5, agent_emb_hidden=16):
        super(DRQNet, self).__init__()
        self.num_actions = num_actions
        self.conv1 = nn.Conv2d(input_dim, hidden_dims[0], 4, padding=1, stride=4)
        self.conv2 = nn.Conv2d(hidden_dims[0], hidden_dims[1], 3, padding=1, stride=2)
        self.conv3 = nn.Conv2d(hidden_dims[1], hidden_dims[2], 3, padding=1, stride=2)
        self.embedding = nn.Linear(agent_emb_dim, agent_emb_hidden)
        #self.lstm_layer = nn.LSTM(input_size=256, hidden_size=256, num_layers=1, batch_first=True)
        self.lstm_layer = nn.LSTMCell(lstm_input, lstm_out)
        self.input_dim = input_dim
        self.lstm_input = lstm_input
        self.lstm_lut = lstm_out

        self.adv = nn.Linear(lstm_out+agent_emb_hidden, num_actions)
        self.val = nn.Linear(lstm_out+agent_emb_hidden, num_actions)
        #self.l1 = nn.Linear(hidden_dims[1]*3*3+agent_emb_hidden, num_actions)

        if torch.cuda.is_available():
            self.dtype = torch.cuda.FloatTensor
        else:
            self.dtype = torch.FloatTensor

    def forward(self, x, id_, hidden_state, cell_state):
        batch_size = x.shape[0]
        t = torch.relu(self.conv1(x))
        t = torch.relu(self.conv2(t))
        t = torch.relu(self.conv3(t))
        t = t.view(batch_size, self.lstm_input)
        h_n, c_n = self.lstm_layer(t, (hidden_state, cell_state))

        emb = self.embedding(id_)
        out = torch.cat([h_n, emb], dim=1)


        adv_out = self.adv(out)
        val_out = self.val(out)
        qval = val_out.expand(batch_size,self.num_actions) + (adv_out - adv_out.mean(dim=1).unsqueeze(dim=1).expand(batch_size,self.num_actions))
        return qval, h_n,c_n

    def init_hidden_states(self,batch_size):
        h = np.zeros((batch_size,256))
        c = np.zeros((batch_size,256))
        return h,c




