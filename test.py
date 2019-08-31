import os, sys
import json

import numpy as np
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from models.QNet import QNet, QNetConv
from agents.DQN import DQN
from agents.DDQN import DDQN
from agents.DRQN import DRQN
from agents.random import Random
import shutil
import argparse
from attrdict import AttrDict
from garl_gym.scenarios.simple_population_dynamics_ga import SimplePopulationDynamicsGA
from garl_gym.scenarios.simple_population_dynamics import SimplePopulationDynamics
from garl_gym.scenarios.complex_population_dynamics import ComplexPopulationDynamics
from garl_gym.scenarios.genetic_population_dynamics import GeneticPopulationDynamics
from utils import str2bool, make_video


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
argparser.add_argument('--config_file', type=str, default=None)
argparser.add_argument('--multiprocessing', type=str2bool, default=False)
argparser.add_argument('--cpu_cores', type=int, default=None)
argparser.add_argument('--predator_increase_prob', type=float, default=None)
argparser.add_argument('--prey_increase_prob', type=float, default=None)
argparser.add_argument('--wall_prob', type=float, default=None)
argparser.add_argument('--env_type', type=str, default=None)
argparser.add_argument('--model_type', type=str, default=None)
argparser.add_argument('--predator_capacity', type=int, default=None)
argparser.add_argument('--prey_capacity', type=int, default=None)
argparser.add_argument('--health_increase_rate', type=int, default=None)
args = argparser.parse_args()

def make_env(env_type, params):
    if env_type == 'simple_population_dynamics_ga':
        return SimplePopulationDynamicsGA(params)
    elif env_type == 'simple_population_dynamics':
        return SimplePopulationDynamics(params)
    elif env_type == 'complex_population_dynamics':
        return ComplexPopulationDynamics(params)
    elif env_type == 'genetic_population_dynamics':
        return GeneticPopulationDynamics(params)

def save_config(params, experiment_id, test_id):
    config_dir = os.path.join('./results', params.env_type, 'exp_{:d}'.format(experiment_id), 'test_logs', str(test_id))
    try:
        os.makedirs(config_dir)
    except:
        shutil.rmtree(config_dir)
        os.makedirs(config_dir) 
    with open(os.path.join(config_dir, 'test_config.txt'), 'w') as f:
        f.write(json.dumps(params))


def load_config(path_prefix):
    if args.model_type == 'Random':
        f = open('configs/random_config.yaml', 'r')
        return AttrDict(yaml.load(f)).parameters
    else:
        params = json.loads(open(os.path.join(path_prefix, 'config.txt')).read())

    return AttrDict(params)

def read_yaml(path):
    f = open(path, 'r')
    return AttrDict(yaml.load(f)).parameters

def ddqn(params, env_type, experiment_id, test_id):
    params['experiment_id'] = experiment_id
    params['test_id'] = test_id
    env = make_env(env_type, params)
    env.make_world(wall_prob=params.wall_prob, food_prob=0)
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
    env.make_world(wall_prob=params.wall_prob, food_prob=0)
    q_net = torch.load(args.model_file).cuda()
    agent = DQN(params,
                env,
                q_net,
                nn.MSELoss(),
                optim.RMSprop)
    agent.test()

def drqn(params, env_type, experiment_id, test_id):
    params['experiment_id'] = experiment_id
    params['test_id'] = test_id

    env = make_env(env_type, params)
    env.make_world(wall_prob=params.wall_prob, food_prob=0)
    q_net = torch.load(args.model_file).cuda()
    agent = DRQN(params,
                env,
                q_net,
                nn.MSELoss(),
                optim.RMSprop)
    agent.test()



def random(params, env_type, experiment_id, test_id):
    params['experiment_id'] = experiment_id
    params['test_id'] = test_id

    env = make_env(env_type, params)
    env.make_world(wall_prob=params.wall_prob, food_prob=0)
    agent = Random(params,
                env)
    agent.test()



if __name__ == '__main__':
    if args.config_file is None:
        params = load_config(args.path_prefix)
    else:
        params = read_yaml(args.config_file)


    if args.model_type == 'Random':
        params['model_type'] = args.model_type

    if args.predator_num is not None:
        params.predator_num = args.predator_num
    if args.prey_num is not None:
        params.prey_num = args.prey_num

    if args.height is not None:
        params.height = args.height
    if args.width is not None:
        params.width = args.width

    if args.multiprocessing and args.cpu_cores is not None:
        params.multiprocessing = args.multiprocessing
        params.cpu_cores = args.cpu_cores
    if args.wall_prob is not None:
        params.wall_prob = args.wall_prob

    if args.env_type is not None:
        params['env_type'] = args.env_type

    if args.predator_increase_prob is not None:
        params.predator_increase_prob = args.predator_increase_prob
    if args.prey_increase_prob is not None:
        params.prey_increase_prob = args.prey_increase_prob

    if args.predator_capacity is not None:
        params.predator_capacity = args.predator_capacity
    if args.prey_capacity is not None:
        params.prey_capacity = args.prey_capacity

    if args.health_increase_rate is not None:
        params.health_increase_rate = args.health_increase_rate



    params.video_flag = args.video_flag
    save_config(params, args.experiment_id, args.test_id)

    print(params)

    if params['model_type'] == 'DDQN':
        ddqn(params, params['env_type'], args.experiment_id, args.test_id)
    elif params['model_type'] == 'DQN':
        dqn(params, params['env_type'], args.experiment_id, args.test_id)
    elif params['model_type'] == 'DRQN':
        drqn(params, params['env_type'], args.experiment_id, args.test_id)
    elif params['model_type']=='Random':
        random(params, args.env_type, args.experiment_id, args.test_id)
    else:
        raise NotImplementedError

    if args.video_flag == True:
        img_dir = os.path.join(args.path_prefix, 'test_images', str(args.test_id))
        st = 0
        ed = len(os.listdir(img_dir))
        images = [os.path.join(img_dir, '{:d}.png'.format(i+1)) for i in range(st, ed)]

        make_video(images, os.path.join(img_dir, 'video.avi'))


