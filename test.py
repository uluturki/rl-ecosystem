import os, sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from models.QNet import QNet
from agents.DQN import DQN
import argparse
from attrdict import AttrDict
from garl_gym.scenarios.simple_population_dynamics import SimplePopulationDynamics


argparser = argparse.ArgumentParser()

argparser.add_argument('--model_file', type=str)
argparser.add_argument('--experiment_num', type=int, default=0)
argparser.add_argument('--test_num', type=int, default=0)

argv = argparser.parse_args()


args = {'predator_num': 500, 'prey_num': 250, 'num_actions': 4, 'height':300, 'damage_per_step': 0.01, 'img_length': 5, 'max_hunt_square': 3, 'max_speed': 1, 'max_acceleration': 1,
        'width': 300, 'batch_size': 512, 'vision_width': 7, 'vision_height': 7, 'max_health': 1.0, 'min_health': 0.5, 'max_crossover': 3, 'wall_prob': 0.02, 'wall_seed': 20, 'food_prob': 0}
        #'width': 70, 'batch_size': 1, 'view_args': ['2500-5-5-0','2500-5-5-1','2500-5-5-2','']}
args = AttrDict(args)

env = SimplePopulationDynamics(args)
env.make_world(wall_prob=0.02, wall_seed=20, food_prob=0)

q_net = torch.load(argv.model_file)
agent = DQN(argv,
            env,
            q_net,
            nn.MSELoss(),
            optim.RMSprop)

agent.test(argv.model_file)
