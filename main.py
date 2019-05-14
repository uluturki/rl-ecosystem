import os
import numpy as np
from attrdict import AttrDict

from garl_gym.scenarios.simple_population_dynamics import SimplePopulationDynamics
import matplotlib.pyplot as plt
import seaborn as sns
from models.QNet import QNet
from agents.DQN import DQN
import argparse
import cv2

import torch
import torch.nn as nn
import torch.optim as optim

argparser = argparse.ArgumentParser()

'''
Training arguments
'''
argparser.add_argument('--time_step', type=int, default=10)
argparser.add_argument('--epochs', type=int, default=100)
argparser.add_argument('--experiment_num', type=int, default=0)

argv = argparser.parse_args()



args = {'predator_num': 500, 'prey_num': 250, 'num_actions': 4, 'height':300, 'damage_per_step': 0.01, 'img_length': 5, 'max_hunt_square': 3, 'max_speed': 1, 'max_acceleration': 1,
        'width': 300, 'batch_size': 512, 'vision_width': 7, 'vision_height': 7, 'max_health': 1.0, 'min_health': 0.5, 'max_crossover': 3, 'wall_prob': 0.02, 'wall_seed': 20, 'food_prob': 0}
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

    #if i % 10 == 0:
    #    predators.append(len(env.predators))
    #    preys.append(len(env.preys))
    #    total_food.append(env.num_food)
    #    total_rewards.append(sum_rewards/10.)
    #    sum_rewards = 0
pdb.set_trace()
sum_rewards += np.sum(rewards)
#env.plot_map_cv2()

if len(env.predators) < 2:
    env.increase_predator(0.2)
elif len(env.preys)<2:
    env.increase_prey(0.2)

#env.crossover_preys(crossover_rate=0.05)
#env.crossover_predators(crossover_rate=0.05)
env.increase_prey(0.03)
env.increase_predator(0.006)
env.increase_food(prob=0.005)

sns.set_style("darkgrid")
plt.plot(list(range(0, i, 10)), predators)
plt.plot(list(range(0, i, 10)), preys)
#plt.plot(list(range(0, i, 10)), total_food)
plt.legend(['predators', 'preys'])
plt.show()
#
#plt.figure()
#plt.plot(total_rewards)
#plt.show()
plt.savefig('dynamics.png')

print(env.predators,env.preys)
