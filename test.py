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
from utils import str2bool


argparser = argparse.ArgumentParser()

argparser.add_argument('--model_file', type=str)
argparser.add_argument('--path_prefix', type=str)
argparser.add_argument('--experiment_id', type=int, default=0)
argparser.add_argument('--test_id', type=int, default=0)
argparser.add_argument('--predator_num', type=int, default=None)
argparser.add_argument('--prey_num', type=int, default=None)
argparser.add_argument('--width', type=int, default=None)
argparser.add_argument('--height', type=int, default=None)
argparser.add_argument('--video_flag', type=str2bool, nargs='?', const=True, default=False)

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
    q_net = torch.load(args.model_file).cuda()
    agent = DDQN(params,
                env,
                q_net,
                nn.MSELoss(),
                optim.RMSprop)
    agent.test()

def dqn(params, env_type, experiment_id, test_id):
    params['experiment_id'] = experiment_id
    params['test_id'] = test_id

    env = make_env(env_type, params)
    env.make_world(wall_prob=0.02, wall_seed=20, food_prob=0)
    q_net = torch.load(args.model_file).cuda()
    agent = DQN(params,
                env,
                q_net,
                nn.MSELoss(),
                optim.RMSprop)
    agent.test()


if __name__ == '__main__':

    params = load_config(args.path_prefix)

    if args.predator_num is not None:
        params.predator_num = args.predator_num
    if args.prey_num is not None:
        params.prey_num = args.prey_num

    if args.height is not None:
        params.height = args.height
    if args.width is not None:
        params.width = args.width

    params.video_flag = args.video_flag

    if params['model_type'] == 'DDQN':
        ddqn(params, params['env_type'], args.experiment_id, args.test_id)
    elif params['model_type'] == 'DQN':
        dqn(params, params['env_type'], args.experiment_id, args.test_id)
    else:
        raise NotImplementedError

