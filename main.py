import os
import numpy as np
from attrdict import AttrDict

from garl_gym.scenarios.simple_population_dynamics_ga_near import SimplePopulationDynamics
import matplotlib.pyplot as plt
import seaborn as sns
from models.QNet import QNet
from agents.DQN import DQN
import argparse
import cv2
import json

import torch
import torch.nn as nn
import torch.optim as optim
import shutil

argparser = argparse.ArgumentParser()

'''
Training arguments
'''
argparser.add_argument('--time_step', type=int, default=10)
argparser.add_argument('--epochs', type=int, default=100)
argparser.add_argument('--experiment_num', type=int, default=0)

argv = argparser.parse_args()



args = {'predator_num': 500, 'prey_num': 250, 'num_actions': 4, 'height':500, 'damage_per_step': 0.01, 'img_length': 5, 'max_hunt_square': 3, 'max_speed': 1, 'max_acceleration': 1,
        'width': 500, 'batch_size': 512, 'vision_width': 7, 'vision_height': 7, 'max_health': 1.0, 'min_health': 0.5, 'max_crossover': 3, 'wall_prob': 0.02, 'wall_seed': 20, 'food_prob': 0}

config_dir = os.path.join('./results', 'exp_{:d}'.format(argv.experiment_num))

try:
    os.makedirs(config_dir)
except:
    shutil.rmtree(config_dir)
    os.makedirs(config_dir)

with open(os.path.join(config_dir, 'config.txt'), 'w') as f:
    f.write(json.dumps(args))


#args = {'predator_num': 20, 'prey_num': 20, 'num_actions': 4, 'height':100, 'damage_per_step': 0.01, 'img_length': 5, 'max_hunt_square': 3, 'max_speed': 1, 'max_acceleration': 1,
#        'width': 100, 'batch_size': 512, 'vision_width': 7, 'vision_height': 7, 'max_health': 1.0, 'min_health': 0.5, 'max_crossover': 3, 'wall_prob': 0.02, 'wall_seed': 20, 'food_prob': 0}
        #'width': 70, 'batch_size': 1, 'view_args': ['2500-5-5-0','2500-5-5-1','2500-5-5-2','']}
args = AttrDict(args)

env = SimplePopulationDynamics(args)
env.make_world(wall_prob=0.02, wall_seed=20, food_prob=0)
#env.plot_map()

predators = []
preys = []
total_rewards = []
total_food = []

sum_rewards = 0

q_net = QNet(args.vision_width*args.vision_height*4+5)
agent = DQN(argv,
            env,
            q_net,
            nn.MSELoss(),
            optim.RMSprop)

agent.train()
