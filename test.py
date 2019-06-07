import os, sys
import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from models.QNet import QNet, QNetConv
from agents.DQN import DQN
from agents.DDQN import DDQN
import argparse
from attrdict import AttrDict
from garl_gym.scenarios.simple_population_dynamics_ga import SimplePopulationDynamicsGA
from garl_gym.scenarios.simple_population_dynamics import SimplePopulationDynamics


argparser = argparse.ArgumentParser()

argparser.add_argument('--model_file', type=str)
argparser.add_argument('--path_prefix', type=str)
argparser.add_argument('--experiment_id', type=int, default=0)
argparser.add_argument('--test_id', type=int, default=0)

args = argparser.parse_args()

def make_env(env_type, params):
    if env_type == 'simple_population_dynamics_ga':
        return SimplePopulationDynamicsGA(params)
    elif env_type == 'simple_population_dynamics':
        return SimplePopulationDynamics(params)


def load_config(path_prefix):
    params = json.loads(open(os.path.join(path_prefix, 'config.txt')).read())
    return AttrDict(params)

def ddqn(params, env_type, experiment_id, test_id):
    params['experiment_id'] = experiment_id
    params['test_id'] = test_id
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
    agent.test(args.model_file)

def dqn(params, env_type, experiment_id, test_id):
    params['experiment_id'] = experiment_id
    params['test_id'] = test_id

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
    agent.test(args.model_file)


if __name__ == '__main__':

    params = load_config(args.path_prefix)

    if params['model_type'] == 'DDQN':
        ddqn(params, params['env_type'], args.experiment_id, args.test_id)
    elif params['model_type'] == 'DQN':
        dqn(params, params['env_type'], args.experiment_id, args.test_id)
    else:
        raise NotImplementedError

