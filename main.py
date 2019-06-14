import os
import numpy as np
from attrdict import AttrDict
import yaml

from garl_gym.scenarios.simple_population_dynamics_ga import SimplePopulationDynamicsGA
from garl_gym.scenarios.simple_population_dynamics_ga_utility import SimplePopulationDynamicsGAUtility
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
from trainer import Trainer

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
    elif env_type == 'simple_population_dynamics_ga_utility':
        return SimplePopulationDynamicsGAUtility(params)





@click.group()
def main():
    pass

@main.command()
@click.option('--env_type', required=True)
@click.option('--experiment_id', help='Experiment Id', required=True, type=int)
@click.option('--config_file', help='config file', type=str, default='./configs/config.yaml')
def ddqn(env_type, experiment_id, config_file):
    params = read_yaml(config_file)
    params['model_type'] = 'DDQN'
    params['env_type'] = env_type
    params['experiment_id'] = experiment_id

    save_config(params, experiment_id)
    env = make_env(env_type, params)
    env.make_world(wall_prob=0.02, wall_seed=20, food_prob=0)
    if params['load_weight'] is None:
        if params['obs_type'] == 'conv':
            q_net = QNetConv(params.input_dim)
        else:
            q_net = QNet(params.vision_width*params.vision_height*4+5)
    else:
        q_net = torch.load(params['load_weight'])
    agent = DDQN(params,
                env,
                q_net,
                nn.MSELoss(),
                optim.RMSprop)
    agent.train(params.episodes,
                params.episode_step,
                params.random_step,
                params.min_greedy,
                params.max_greedy,
                params.greedy_step,
                params.update_period)


@main.command()
@click.option('--env_type', required=True)
@click.option('--experiment_id', help='Experiment Id', required=True, type=int)
@click.option('--config_file', help='config file', type=str, default='./configs/config.yaml')
def dqn(env_type, experiment_id, config_file):
    params = read_yaml(config_file)
    params['model_type'] = 'DQN'
    params['env_type'] = env_type
    params['experiment_id'] = experiment_id

    save_config(params, experiment_id)
    env = make_env(env_type, params)
    env.make_world(wall_prob=0.02, wall_seed=20, food_prob=0)
    if params['load_weight'] is None:
        if params['obs_type'] == 'conv':
            q_net = QNetConv(params.input_dim)
        else:
            q_net = QNet(params.vision_width*params.vision_height*4+5)
    else:
        q_net = torch.load(params['load_weight'])
    agent = DQN(params,
                env,
                q_net,
                nn.MSELoss(),
                optim.RMSprop)
    agent.train(params.episodes,
                params.episode_step,
                params.random_step,
                params.min_greedy,
                params.max_greedy,
                params.greedy_step,
                params.update_period)

@main.command(name='dqn_two_agents')
@click.option('--env_type', required=True)
@click.option('--experiment_id', help='Experiment Id', required=True, type=int)
@click.option('--config_file', help='config file', type=str, default='./configs/config.yaml')
def dqn_two_agents(env_type, experiment_id, config_file):
    params = read_yaml(config_file)
    params['model_type'] = 'DQN'
    params['env_type'] = env_type
    params['experiment_id'] = experiment_id

    save_config(params, experiment_id)
    env = make_env(env_type, params)
    env.make_world(wall_prob=0.02, wall_seed=20, food_prob=0)
    if params['load_weight'] is None:
        if params['obs_type'] == 'conv':
            q_net = QNetConv(params.input_dim)
        else:
            q_net = QNet(params.vision_width*params.vision_height*4+5)
    else:
        q_net = torch.load(params['load_weight'])
    agent_predator = DQN(params,
                env,
                q_net,
                nn.MSELoss(),
                optim.RMSprop)
    agent_prey = DQN(params,
                env,
                q_net,
                nn.MSELoss(),
                optim.RMSprop)

    trainer = Trainer(params, env)
    trainer.train(agent_predator, agent_prey)

if __name__ == '__main__':
    main()

