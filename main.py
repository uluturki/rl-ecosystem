import os
import numpy as np
from attrdict import AttrDict

from garl_gym.scenarios.simple_population_dynamics_ga_near import SimplePopulationDynamics
import matplotlib.pyplot as plt
import seaborn as sns
from models.QNet import QNet
from agents.DQN import DQN
#from agents.DDQN import DDQN
import argparse
import cv2
import json
import click

import torch
import torch.nn as nn
import torch.optim as optim
import shutil

def read_yaml(path):
    f = open(path)
    return AttrDict(yaml.load(f))

def save_config(params, experiment_id):
    config_dir = os.path.join('./results', 'exp_{:d}'.format(experiment_id))
    try:
        os.makedirs(config_dir)
    except:
        shutil.rmtree(config_dir)
        os.makedirs(config_dir)

    with open(os.path.join(config_dir, 'config.txt'), 'w') as f:
        f.write(json.dumps(params))


params = read_yaml('config.yaml')



@click.group
def main():
    pass

@main.command()
@click.option('--env', required=True)
@click.option('--experiment_id', help='Experiment Id', required=True)
def double_dqn(env, experiment_id):
    params['model_type'] = 'DDQN'
    save_config(params, experiment_id)
    env = SimplePopulationDynamics(args)
    env.make_world(wall_prob=0.02, wall_seed=20, food_prob=0)

@main.command()
@click.option('--env', required=True)
@click.option('--experiment_id', help='Experiment Id', required=True)
def dqn(env, experiment_id):
    params['model_type'] = 'DQN'
    save_config(params, experiment_id)
    env = SimplePopulationDynamics(args)
    env.make_world(wall_prob=0.02, wall_seed=20, food_prob=0)

if __name__ == '__main__':
    main()

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
