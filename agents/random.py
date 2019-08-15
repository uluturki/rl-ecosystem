import os, sys

import numpy as np
import torch.nn.functional as F
from models import QNet
from torch.autograd import Variable
from copy import deepcopy
from torch.nn.utils.clip_grad import clip_grad_norm
from tqdm import tqdm
import gc
#from torch.utils.tensorboard import SummaryWriter
import shutil
from utils import plot_dynamics, plot_diversity
from garl_gym import scenarios


class Random(object):
    def __init__(self, args, env):
        self.args = args
        self.obs_type = args.obs_type
        self.env = env

        self.num_actions = args.num_actions
        self.video_flag= args.video_flag

    def test(self, test_step=200000):
        if self.args.env_type == 'simple_population_dynamics':
            get_obs = scenarios.simple_population_dynamics.get_obs
        elif self.args.env_type == 'simple_population_dynamics_ga':
            get_obs = scenarios.simple_population_dynamics_ga.get_obs
        elif self.args.env_type == 'simple_population_dynamics_ga_utility':
            get_obs = scenarios.simple_population_dynamics_ga_utility.get_obs
        elif self.args.env_type == 'simple_population_dynamics_ga_action':
            get_obs = scenarios.simple_population_dynamics_ga_action.get_obs
            assert self.num_actions == 5, 'Expected the number of actions 5, but got {:d}'.format(self.num_actions)
        elif self.args.env_type == 'complex_population_dynamics':
            get_obs = scenarios.complex_population_dynamics.get_obs
        elif self.args.env_type == 'genetic_population_dynamics':
            get_obs = scenarios.genetic_population_dynamics.get_obs

        total_reward = 0
        bar = tqdm()


        img_dir = os.path.join('results', self.args.env_type, 'exp_{:d}'.format(self.args.experiment_id), 'test_images', str(self.args.test_id))
        log_dir = os.path.join('results', self.args.env_type, 'exp_{:d}'.format(self.args.experiment_id), 'test_logs', str(self.args.test_id))

        try:
            os.makedirs(img_dir)
        except:
            shutil.rmtree(img_dir)
            os.makedirs(img_dir)
        #try:
        #    os.makedirs(log_dir)
        #except:
        #    shutil.rmtree(log_dir)
        #    os.makedirs(log_dir)
        log = open(os.path.join(log_dir, 'log.txt'), 'w')

        timesteps = 0

        for i in range(test_step):
            if self.video_flag:
                self.env.dump_image(os.path.join(img_dir, '{:d}.png'.format(timesteps+1)))

            if self.args.env_type == 'simple_population_dynamics_ga_utility' and i % 200 == 0:
                plot_diversity(self.env.predators.values(), self.env.preys.values(), log_dir, i)

            if i % 100 == 0:
                plot_dynamics(os.path.join(log_dir, 'log.txt'), 0)


            obs = get_obs(self.env, only_view=True)
            actions = {}
            for id, agent in self.env.agents.items():
                actions[id] = np.random.randint(self.num_actions)
            self.env.take_actions(actions)
            next_view_batches, rewards, killed = get_obs(self.env)
            self.env.killed = killed
            episode_reward = np.sum(list(rewards.values()))
            increase_predators = self.env.increase_predators
            increase_preys = self.env.increase_preys
            killed = self.env.remove_dead_agents()

            msg = "episode step {:03d}".format(i)
            bar.set_description(msg)
            bar.update(1)

            info = "Step\t{:03d}\tReward\t{:5.3f}\tnum_agents\t{:d}\tnum_preys\t{:d}\tnum_predators\t{:d}\tincrease_predators\t{:d}\tincrease_preys\t{:d}".format(i, episode_reward/len(obs), len(self.env.agents), len(self.env.preys), len(self.env.predators), increase_predators, increase_preys)
            timesteps += 1

            if self.args.env_type == 'simple_population_dynamics':
                if i % self.args.increase_every == 0:
                    self.env.increase_prey(self.args.prey_increase_prob)
                    self.env.increase_predator(self.args.predator_increase_prob)
            elif self.args.env_type == 'complex_population_dynamics':
                info = "Step\t{:03d}\tReward\t{:5.3f}\tnum_agents\t{:d}\tnum_preys\t{:d}\tnum_predators\t{:d}\tincrease_predators\t{:d}\tincrease_preys\t{:d}\tkilled_agents\t{:d}\taverage_predator_health\t{:5.3f}\taverage_prey_health\t{:5.3f}\taverage_predator_attack\t{:5.3f}\taverage_prey_attack\t{:5.3f}\taverage_predator_resilience\t{:5.3f}\taverage_prey_resilience\t{:5.3f}\taverage_predator_speed\t{:5.3f}\taverage_prey_speed\t{:5.3f}".format(i, episode_reward/len(obs), len(self.env.agents), len(self.env.preys), len(self.env.predators),increase_predators, increase_preys, len(killed), np.mean(self.env.predator_health), np.mean(self.env.prey_health), np.mean(self.env.predator_attack), np.mean(self.env.prey_attack), np.mean(self.env.predator_resilience), np.mean(self.env.prey_resilience), np.mean(self.env.predator_speed), np.mean(self.env.prey_speed))
                self.env.crossover_prey(self.args.crossover_scope, crossover_rate=self.args.prey_increase_prob, mutation_prob=self.args.mutation_prob)
                self.env.crossover_predator(self.args.crossover_scope, crossover_rate=self.args.predator_increase_prob, mutation_prob=self.args.mutation_prob)
                if i % self.args.increase_every == 0:
                    self.env.add_preys(1)
                    self.env.add_predators(1)
            elif self.args.env_type == 'genetic_population_dynamics':
                info = "Step\t{:03d}\tReward\t{:5.3f}\tnum_agents\t{:d}\tnum_preys\t{:d}\tnum_predators\t{:d}\tincrease_predators\t{:d}\tincrease_preys\t{:d}\tkilled_agents\t{:d}\taverage_predator_health\t{:5.3f}\taverage_prey_health\t{:5.3f}\taverage_predator_attack\t{:5.3f}\taverage_prey_attack\t{:5.3f}\taverage_predator_resilience\t{:5.3f}\taverage_prey_resilience\t{:5.3f}\taverage_predator_speed\t{:5.3f}\taverage_prey_speed\t{:5.3f}\tvar_predator_health\t{:5.3f}\tvar_prey_health\t{:5.3f}\tvar_predator_attack\t{:5.3f}\tvar_prey_attack\t{:5.3f}\tvar_predator_resilience\t{:5.3f}\var_prey_resilience\t{:5.3f}\tvar_predator_speed\t{:5.3f}\tvar_prey_speed\t{:5.3f}".format(i, episode_reward/len(obs), len(self.env.agents), len(self.env.preys), len(self.env.predators),increase_predators, increase_preys, len(killed), np.mean(self.env.predator_health), np.mean(self.env.prey_health), np.mean(self.env.predator_attack), np.mean(self.env.prey_attack), np.mean(self.env.predator_resilience), np.mean(self.env.prey_resilience), np.mean(self.env.predator_speed), np.mean(self.env.prey_speed), np.var(self.env.predator_health), np.var(self.env.prey_health), np.var(self.env.predator_attack), np.var(self.env.prey_attack), np.var(self.env.predator_resilience), np.var(self.env.prey_resilience), np.var(self.env.predator_speed), np.var(self.env.prey_speed))
                self.env.crossover_prey(self.args.prey_increase_prob, mutation_prob=self.args.mutation_prob)
                self.env.crossover_predator(self.args.predator_increase_prob, mutation_prob=self.args.mutation_prob)
            elif self.args.env_type != 'simple_population_dynamics_ga_action':
                self.env.crossover_prey(self.args.crossover_scope, crossover_rate=self.args.prey_increase_prob)
                self.env.crossover_predator(self.args.crossover_scope, crossover_rate=self.args.predator_increase_prob)
                self.env.add_preys(1)
                #if len(self.env.predators) < 200:
                self.env.add_predators(1)

            log.write(info+'\n')
            log.flush()

            if len(self.env.predators) < 1 or len(self.env.preys) < 1 or len(self.env.predators) > 20000 or len(self.env.preys) > 20000:
                log.close()
                break
        #images = [os.path.join(img_dir, ("{:d}.png".format(j+1))) for j in range(timesteps)]
        #self.env.make_video(images, outvid=os.path.join(img_dir, 'episode_{:d}.avi')


