import os, sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class DQN(nn.Module):
    def __init__(self, input_dim, hidden_dims=[32, 32], action_size=4, agent_emb_dim=5):
        super(DQN, self).__init__()
        self.agent_emb_dim = agent_emb_dim
        self.agent_embeddings = {}

        self.num_actions = action_size
        self.l1 = nn.Linear(54, hidden_dims[0])
        self.l2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.l3 = nn.Linear(hidden_dims[1], action_size)

        if torch.cuda.is_available():
            self.dtype = torch.cuda.FloatTensor
        else:
            self.dtype = torch.FloatTensor

    def forward(self, x):
        t = F.sigmoid(self.l1(x))
        t = F.sigmoid(self.l2(t))
        t = F.sigmoid(self.l3(t))
        return t

    def train(self, view_batches, action_batches, rewards, maxQ_batches, learning_rate=0.001):
        def split_id_value(input_):
            ret_id = []
            ret_value = []
            for item in input_:
                ret_id.append(item[0])
                ret_value.append(item[1])
            return ret_id, ret_value
        for i in range(len(view_batches)):
            view_id, view_values = self.process_view_with_emb_batch(view_batches[i])
            action_id, action_value = split_id_value(actions_batches[i])
            maxQ_id, maxQ_value = split_id_value(maxQ_batches[i])
            assert view_id == action_id == maxQ_id
            reward_value = []
            for id in view_id:
                if id in rewards:
                    reward_value.append(rewards[id])
                else:
                    reward_value.append(0.)
            out = self(view_values)




    def infer_actions(self, view_batches, epsilon=0.1):
        ret_actions = []
        ret_actions_batch = []

        for view in view_batches:
            batch_id, batch_view = self.process_view_with_emb_batch(view)
            actions_batch = self._inference(batch_view, epsilon=epsilon)
            action_batch_set = zip(batch_id, actions_batch)
            ret_actions_batch.append(action_batch_set)
            ret_actions.extend(action_batch_set)

        return ret_actions, ret_actions_batch

    def _inference(self, batch_view, epsilon):
        value_s_a = self.forward(batch_view)
        all_actions = range(self.num_actions)
        actions = []
        for i in range(len(value_s_a)):
            if np.random.rand() < epsilon:
                actions.append(np.random.choice(all_actions))
            else:
                actions.append(np.argmax(value_s_a[i]))
        return np.array(actions)

    def process_view_with_emb_batch(self, input_view):
        batch_id = []
        batch_view = []
        for id, view in input_view:
            if id in self.agent_embeddings:
                new_view = np.concatenate((self.agent_embeddings[id], view), 0)
                batch_view.append(new_view)
            else:
                new_embedding = np.random.normal(size=[self.agent_emb_dim])
                self.agent_embeddings[id] = new_embedding
                new_view = np.concatenate((new_embedding, view), 0)
                batch_view.append(new_view)
        return batch_id, Variable(torch.from_numpy(np.array(batch_view))).type(self.dtype)


