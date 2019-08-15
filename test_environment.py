import os
import numpy as np
from attrdict import AttrDict
import yaml

from garl_gym.scenarios.simple_population_dynamics_ga import SimplePopulationDynamicsGA
from garl_gym.scenarios.simple_population_dynamics_ga_action import SimplePopulationDynamicsGAAction
from garl_gym.scenarios.simple_population_dynamics_ga_utility import SimplePopulationDynamicsGAUtility
from garl_gym.scenarios.simple_population_dynamics import SimplePopulationDynamics
from garl_gym.scenarios.simple_population_dynamics_rule_base import SimplePopulationDynamicsRuleBase
from garl_gym.scenarios.complex_population_dynamics import ComplexPopulationDynamics
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import cv2
import json
import click
from utils import str2bool

import torch
import torch.nn as nn
import torch.optim as optim
import shutil
from trainer import Trainer

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
argparser.add_argument('--config_file', type=str, default='./configs/config.yaml')
argparser.add_argument('--multiprocessing', type=str2bool, default=False)
argparser.add_argument('--cpu_cores', type=int, default=None)
argparser.add_argument('--predator_increase_prob', type=float, default=None)
argparser.add_argument('--prey_increase_prob', type=float, default=None)
argparser.add_argument('--wall_prob', type=float, default=None)
argparser.add_argument('--env_type', type=str, default=None, required=True)
argparser.add_argument('--model_type', type=str, default=None)
argparser.add_argument('--predator_capacity', type=int, default=None)
argparser.add_argument('--prey_capacity', type=int, default=None)
argparser.add_argument('--health_increase_rate', type=int, default=None)
args = argparser.parse_args()

def read_yaml(path):
    f = open(path, 'r')
    return AttrDict(yaml.load(f)).parameters

def make_env(env_type, params):
    if env_type == 'simple_population_dynamics_ga':
        return SimplePopulationDynamicsGA(params)
    elif env_type == 'simple_population_dynamics':
        return SimplePopulationDynamics(params)
    elif env_type == 'simple_population_dynamics_ga_utility':
        return SimplePopulationDynamicsGAUtility(params)
    elif env_type == 'simple_population_dynamics_ga_action':
        return SimplePopulationDynamicsGAAction(params)
    elif env_type == 'complex_population_dynamics':
        return ComplexPopulationDynamics(params)

params = read_yaml(args.config_file)

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


env = make_env(args.env_type, params)
env.make_world(wall_prob=params.wall_prob, food_prob=0)



for i in range(10000):

    num_agents = len(env.agents)
    actions = np.random.randint(4, size=num_agents)
    actions = dict(zip(list(env.agents.keys()), actions))
    env.take_actions(actions)

    env.crossover_prey(params.crossover_scope, crossover_rate=params.prey_increase_prob, mutation_prob=params.mutation_prob)
    env.crossover_predator(params.crossover_scope, crossover_rate=params.predator_increase_prob, mutation_prob=params.mutation_prob)

