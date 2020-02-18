import os
import numpy as np
from attrdict import AttrDict
import yaml

from garl_gym.scenarios.simple_population_dynamics_ga import SimplePopulationDynamicsGA
from garl_gym.scenarios.simple_population_dynamics import SimplePopulationDynamics
from garl_gym.scenarios.complex_population_dynamics import ComplexPopulationDynamics
from garl_gym.scenarios.genetic_population_dynamics import GeneticPopulationDynamics
import matplotlib.pyplot as plt
import seaborn as sns
from models.QNet import QNet, QNetConv
from models.DRQNet import DRQNet
from agents.DQN import DQN
from agents.DDQN import DDQN
from agents.DRQN import DRQN
from agents.rule_base import run_rulebase
import argparse
import cv2
import json
import click

import torch
import torch.nn as nn
import torch.optim as optim
import shutil
from tmp.trainer import Trainer

def read_yaml(path):
    f = open(path, 'r')
    return AttrDict(yaml.load(f)).parameters

def save_config(params, experiment_id):
    config_dir = os.path.join('./results', params.env_type, 'exp_{:d}'.format(experiment_id))
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
    elif env_type == 'complex_population_dynamics':
        return ComplexPopulationDynamics(params)
    elif env_type == 'genetic_population_dynamics':
        return GeneticPopulationDynamics(params)

def create_nn(params):
    if params['load_weight'] is None:
        #if params['obs_type'] == 'conv':
        #    q_net = QNetConv(params.input_dim, hidden_dims=params.hidden_dims, num_actions=params.num_actions)
        if params['model_type'] == 'DRQN':
            q_net = DRQNet(params.input_dim, params.lstm_input, params.lstm_out, hidden_dims=params.hidden_dims, num_actions=params.num_actions, agent_emb_dim=params.agent_emb_dim)
        elif params['obs_type'] == 'conv':
            q_net = QNetConv(params.input_dim, hidden_dims=params.hidden_dims, num_actions=params.num_actions, agent_emb_dim=params.agent_emb_dim)
        else:
            q_net = QNet(params.vision_width*params.vision_height*4+5, hidden_dims=params.hidden_dims, num_actions=params.num_actions)
    else:
        q_net = torch.load(params['load_weight'])
    return q_net





@click.group()
def main():
    pass

@main.command()
@click.option('--env_type', required=True)
@click.option('--experiment_id', help='Experiment Id', required=True, type=int)
@click.option('--config_file', help='config file', type=str, default='./configs/config.yaml')
def ddqn(env_type, experiment_id, config_file):
    '''
    Double Deep Q-learning

    Args:
        env_type: Evnrionment Type
        experiment_id: Id for the experiment
        config_file: Path of the config file
    '''

    params = read_yaml(config_file)
    params['model_type'] = 'DDQN'
    params['env_type'] = env_type
    params['experiment_id'] = experiment_id

    save_config(params, experiment_id)
    env = make_env(env_type, params)
    env.make_world(wall_prob=params.wall_prob, wall_seed=20, food_prob=0)
    q_net = create_nn(params)
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
    '''
    Deep Q-learning

    Args:
        env_type: Evnrionment Type
        experiment_id: Id for the experiment
        config_file: Path of the config file
    '''

    params = read_yaml(config_file)
    params['model_type'] = 'DQN'
    params['env_type'] = env_type
    params['experiment_id'] = experiment_id

    save_config(params, experiment_id)
    env = make_env(env_type, params)
    env.make_world(wall_prob=params.wall_prob, wall_seed=20, food_prob=0)
    q_net = create_nn(params)
    agent = DQN(params,
                env,
                q_net,
                nn.MSELoss(),
                optim.RMSprop)
    agent.train(params.episodes,
                params.episode_step,
                params.random_step, params.min_greedy, params.max_greedy, params.greedy_step,
                params.update_period)

@click.option('--config_file', help='config file', type=str, default='./configs/config.yaml')
def dqn_two_agents(env_type, experiment_id, config_file):
    params = read_yaml(config_file)
    params['model_type'] = 'DQN'
    params['env_type'] = env_type
    params['experiment_id'] = experiment_id
    save_config(params, experiment_id)
    env = make_env(env_type, params)
    env.make_world(wall_prob=params.wall_prob, wall_seed=20, food_prob=0)
    q_net = create_nn(params)
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


@main.command(name='drqn')
@click.option('--env_type', required=True)
@click.option('--experiment_id', help='Experiment Id', required=True, type=int)
@click.option('--config_file', help='config file', type=str, default='./configs/config.yaml')
def drqn(env_type, experiment_id, config_file):
    '''
    Deep Recurrent Q-learning

    Args:
        env_type: Evnrionment Type
        experiment_id: Id for the experiment
        config_file: Path of the config file
    '''

    params = read_yaml(config_file)
    params['model_type'] = 'DRQN'
    params['env_type'] = env_type
    params['experiment_id'] = experiment_id

    save_config(params, experiment_id)
    env = make_env(env_type, params)
    env.make_world(wall_prob=params.wall_prob, food_prob=0)
    q_net = create_nn(params)
    agent = DRQN(params,
                env,
                q_net,
                nn.MSELoss(),
                optim.RMSprop)
    agent.train(params.episodes,
                params.episode_step,
                params.random_step, params.min_greedy, params.max_greedy, params.greedy_step,
                params.update_period)



if __name__ == '__main__':
    main()

