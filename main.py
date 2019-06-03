import os
import numpy as np
from attrdict import AttrDict
import yaml

from garl_gym.scenarios.simple_population_dynamics_ga import SimplePopulationDynamicsGA
from garl_gym.scenarios.simple_population_dynamics import SimplePopulationDynamics
import matplotlib.pyplot as plt
import seaborn as sns
from models.QNet import QNet, QNetConv
from agents.DQN import DQN
from agents.DDQN import DDQN
import argparse
import cv2
import json
import click

import torch
import torch.nn as nn
import torch.optim as optim
import shutil

def read_yaml(path):
    f = open(path, 'r')
    return AttrDict(yaml.load(f)).parameters

def save_config(params, experiment_id):
    config_dir = os.path.join('./results', 'exp_{:d}'.format(experiment_id))
    try:
        os.makedirs(config_dir)
    except:
        shutil.rmtree(config_dir)
        os.makedirs(config_dir) 
    with open(os.path.join(config_dir, 'config.txt'), 'w') as f:
        f.write(json.dumps(params))

def make_env(env_type, params):
    if env_type == 'simple_population_dynamics_ga':
        return SimplePopulationDynamicsGA(params)
    elif env_type == 'simple_population_dynamics':
        return SimplePopulationDynamics(params)


params = read_yaml('config.yaml')
import pdb
pdb.set_trace()



@click.group()
def main():
    pass

@main.command()
@click.option('--env_type', required=True)
@click.option('--experiment_id', help='Experiment Id', required=True, type=int)
def ddqn(env_type, experiment_id):
    params['model_type'] = 'DDQN'
    params['env_type'] = env_type
    params['experiment_id'] = experiment_id

    save_config(params, experiment_id)
    env = make_env(env_type, params)
    env.make_world(wall_prob=0.02, wall_seed=20, food_prob=0)
    if params['obs_type'] == 'conv':
        q_net = QNetConv(params.input_dim)
    else:
        q_net = QNet(params.vision_width*params.vision_height*4+5)
    agent = DDQN(params,
                env,
                q_net,
                nn.MSELoss(),
                optim.RMSprop)
    agent.train()


@main.command()
@click.option('--env_type', required=True)
@click.option('--experiment_id', help='Experiment Id', required=True, type=int)
def dqn(env_type, experiment_id):
    params['model_type'] = 'DQN'
    params['env_type'] = env_type
    params['experiment_id'] = experiment_id

    save_config(params, experiment_id)
    env = make_env(env_type, params)
    env.make_world(wall_prob=0.02, wall_seed=20, food_prob=0)
    if params['obs_type'] == 'conv':
        q_net = QNetConv(params.input_dim)
    else:
        q_net = QNet(params.vision_width*params.vision_height*4+5)
    agent = DQN(params,
                env,
                q_net,
                nn.MSELoss(),
                optim.RMSprop)
    agent.train()

if __name__ == '__main__':
    main()

