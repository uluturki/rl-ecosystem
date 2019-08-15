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
from garl_gym.scenarios.simple_population_dynamics_ga_action import SimplePopulationDynamicsGAAction
from garl_gym.scenarios.simple_population_dynamics_ga_utility import SimplePopulationDynamicsGAUtility
from garl_gym.scenarios.simple_population_dynamics import SimplePopulationDynamics
from garl_gym.scenarios.complex_population_dynamics import ComplexPopulationDynamics
from utils import str2bool, make_video
from tqdm import tqdm
import optuna
import pandas as pd


argparser = argparse.ArgumentParser()
argparser.add_argument('--path_prefix', type=str)
argparser.add_argument('--model_type', type=str, default=None)
argparser.add_argument('--model_file', type=str)
argparser.add_argument('--experiment_id', type=int)
argparser.add_argument('--test_id', type=int)
args = argparser.parse_args()

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

def load_config(path_prefix):
    if args.model_type == 'Random':
        f = open('configs/random_config.yaml', 'r')
        return AttrDict(yaml.load(f)).parameters
    else:
        params = json.loads(open(os.path.join(path_prefix, 'config.txt')).read())

    return AttrDict(params)

global_predator_mse_list = []
global_prey_mse_list = []
df = pd.read_csv('./data/hare_lynx_data.csv')

hare = np.array(df['Hare'])*100
lynx = np.array(df['Lynx'])*100

params = load_config(args.path_prefix)
params.predator_num = 3000
params.prey_num = 2000

log_dir = os.path.join('results', params.env_type, 'exp_{:d}'.format(args.experiment_id), 'test_logs', str(args.test_id))

try:
    os.makedirs(log_dir)
except:
    shutil.rmtree(log_dir)
    os.makedirs(log_dir)



q_net = torch.load(args.model_file).cuda()
def objective(trial):
    predator_mse_list = []
    prey_mse_list = []
    points = 25
    #T = trial.suggest_discrete_uniform('timestep', 300, 550, 50)
    T = trial.suggest_discrete_uniform('timestep', 300, 600, 50)
    predator_increase_rate = trial.suggest_uniform('predator_increase_rate', 0.001, 0.02)
    prey_increase_rate = trial.suggest_uniform('prey_increase_rate', 0.001, 0.02)
    health_increase_rate = trial.suggest_int('predator_health_increase_rate', 1, 4)
    height = trial.suggest_discrete_uniform('height', 500, 1000, 100)
    width = height
    params.predator_increase_prob = predator_increase_rate
    params.prey_increase_prob = prey_increase_rate
    params.width = int(width)
    params.height = int(height)
    params.test_id = args.test_id
    params.experiment_id = args.experiment_id
    params.health_increase_rate = health_increase_rate
    params.predator_capacity = 25000
    params.prey_capacity = 25000
    params.cpu_cores = 4
    params.batch_size = 1024
    params.multiprocessing = True

    env = make_env(params.env_type, params)
    env.make_world(wall_prob=params.wall_prob, food_prob=0)

    agent = DRQN(params,
                env,
                q_net,
                nn.MSELoss(),
                optim.RMSprop)

    step = 0
    env.reset()
    bar = tqdm(range(int(points*T)))
    total_predator_mse = 0
    total_prey_mse = 0

    point = 0

    log = open(os.path.join(log_dir, 'log_trial_{}.txt'.format(str(trial.number))), 'w')

    for t in range(int(points*T)):
        agent.one_iteration(t)

        if (t+1)%T == 0:
            predator_num = agent.env.predator_num
            prey_num = agent.env.prey_num
            target_predator_num = lynx[point]
            target_prey_num = hare[point]

            predator_mse = np.mean((predator_num-target_predator_num)**2)
            prey_mse = np.mean((prey_num-target_prey_num)**2)
            total_predator_mse += predator_mse
            total_prey_mse += prey_mse
            predator_mse_list.append(predator_mse)
            prey_mse_list.append(prey_mse)
            point += 1
        info = "Step\t{:03d}\tnum_agents\t{:d}\tnum_preys\t{:d}\tnum_predators\t{:d}\tincrease_predators\t{:d}\tincrease_preys\t{:d}".format(t, len(env.agents), len(env.preys), len(env.predators), env.increase_predators, env.increase_preys)
        log.write(info+'\n')
        log.flush()
        bar.update(1)

        if len(agent.env.predators) > 16000 or len(agent.env.preys) > 16000:
            total_predator_mse = np.inf
            total_prey_mse = np.inf
            break

    print('predator_mse: {} prey_mse: {}'.format(total_predator_mse, total_prey_mse))
    error = total_predator_mse/points+total_prey_mse/points

    log.close()
    bar.update(0)
    bar.refresh()
    bar.close()
    return error



study = optuna.create_study()
study.optimize(objective, n_trials=100)
