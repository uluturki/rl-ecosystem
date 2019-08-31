import os, sys

import numpy as np
import torch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import QNet
from torch.autograd import Variable
from copy import deepcopy
from torch.nn.utils.clip_grad import clip_grad_norm
from tqdm import tqdm
import gc
#from torch.utils.tensorboard import SummaryWriter
import shutil
from utils import plot_dynamics
from garl_gym import scenarios



class Trainer():
    def __init__(self, args, env):
        self.args = args
        self.env = env
        self.num_actions = args.num_actions
        self.obs_type = args.obs_type

    def train(self, agent_prey, agent_predator):
        eps_greedy = self.args.min_greedy
        g_step = (self.args.max_greedy-self.args.min_greedy) / self.args.greedy_step


        if self.args.env_type == 'simple_population_dynamics':
            get_obs = scenarios.simple_population_dynamics.get_obs
        elif self.args.env_type == 'simple_population_dynamics_ga':
            get_obs = scenarios.simple_population_dynamics_ga.get_obs
        elif self.args.env_type == 'simple_population_dynamics_ga_utility':
            get_obs = scenarios.simple_population_dynamics_ga_utility.get_obs

        #g_step = 0
        #eps_greedy = 0.9

        rounds = 0
        model_dir = os.path.join('results', 'exp_{:d}'.format(self.args.experiment_id), 'models')
        try:
            os.makedirs(model_dir)
        except:
            shutil.rmtree(model_dir)
            os.makedirs(model_dir)



        for episode in range(self.args.episodes):
            loss = 0
            predator_total_reward = 0
            prey_total_reward = 0
            bar = tqdm()


            if episode==0 or len(self.env.predators) < 2 or len(self.env.preys) < 2 or len(self.env.preys) > 10000 or len(self.env.predators) > 10000:
                obs = self.env.reset()

                img_dir = os.path.join('results', 'exp_{:d}'.format(self.args.experiment_id), 'images',str(rounds))
                log_dir = os.path.join('results', 'exp_{:d}'.format(self.args.experiment_id), 'logs', str(rounds))
                try:
                    os.makedirs(img_dir)
                except:
                    shutil.rmtree(img_dir)
                    os.makedirs(img_dir)
                try:
                    os.makedirs(log_dir)
                except:
                    shutil.rmtree(log_dir)
                    os.makedirs(log_dir)
                log = open(os.path.join(log_dir, 'log.txt'), 'w')
                rounds += 1
                timesteps = 0

            for i in range(self.args.episode_step):
                predator_episode_reward = 0
                prey_episode_reward = 0
                if self.args.video_flag:
                    self.env.dump_image(os.path.join(img_dir, '{:d}.png'.format(timesteps+1)))

                eps_greedy += g_step
                eps_greedy = np.clip(eps_greedy, self.args.min_greedy, self.args.max_greedy)


                predator_actions = []
                prey_actions = []
                predator_ids = []
                prey_ids = []
                predator_action_batches = []
                prey_action_batches = []

                predator_obs = get_obs(self.env, only_view=True, predator_obs=True)
                prey_obs = get_obs(self.env, only_view=True, prey_obs=True)
                predator_view_batches = []
                prey_view_batches = []
                predator_view_ids = []
                prey_view_ids = []

                predator_view_values_list = []
                prey_view_values_list = []

                for j in range(len(predator_obs)//self.args.batch_size+1):
                    view = predator_obs[j*self.args.batch_size:(j+1)*self.args.batch_size]
                    if len(view) == 0:
                        continue
                    batch_id, batch_view = agent_predator.process_view_with_emb_batch(view)
                    if np.random.rand() < eps_greedy:
                        action = agent_predator.q_net(batch_view).max(1)[1].cpu().numpy()
                    else:
                        action = np.random.randint(self.num_actions, size=len(batch_view))
                    predator_ids.extend(batch_id)
                    predator_actions.extend(action)
                    predator_action_batches.append(action)
                    predator_view_batches.append(view)
                    predator_view_ids.append(batch_id)
                    predator_view_values_list.append(batch_view)
                predator_num_batches = j

                for j in range(len(prey_obs)//self.args.batch_size+1):
                    view = prey_obs[j*self.args.batch_size:(j+1)*self.args.batch_size]
                    if len(view) == 0:
                        continue
                    batch_id, batch_view = agent_prey.process_view_with_emb_batch(view)
                    if np.random.rand() < eps_greedy:
                        action = agent_prey.q_net(batch_view).max(1)[1].cpu().numpy()
                    else:
                        action = np.random.randint(self.num_actions, size=len(batch_view))
                    prey_ids.extend(batch_id)
                    prey_actions.extend(action)
                    prey_action_batches.append(action)
                    prey_view_batches.append(view)
                    prey_view_ids.append(batch_id)
                    prey_view_values_list.append(batch_view)
                prey_num_batches = j

                predator_actions = dict(zip(predator_ids, predator_actions))
                prey_actions = dict(zip(prey_ids, prey_actions))

                self.env.take_actions(predator_actions)
                self.env.take_actions(prey_actions)

                predator_next_view_batches, predator_rewards, predator_killed = get_obs(self.env, predator_obs=True)
                prey_next_view_batches, prey_rewards, prey_killed = get_obs(self.env, prey_obs=True)
                predator_killed.extend(prey_killed)
                self.env.killed = predator_killed
                predator_total_reward += np.sum(list(predator_rewards.values()))
                prey_total_reward += np.sum(list(prey_rewards.values()))

                predator_episode_reward += np.sum(list(predator_rewards.values()))
                prey_episode_reward += np.sum(list(prey_rewards.values()))

                loss_batch = 0

                for j in range(predator_num_batches):
                    view_id = predator_view_ids[j]
                    view_values = predator_view_values_list[j]
                    next_view = predator_obs[j*self.args.batch_size:(j+1)*self.args.batch_size]
                    next_view_id, next_view_values = agent_predator.process_view_with_emb_batch(next_view)
                    l = agent_predator.update(view_values, predator_action_batches[j], next_view_values, view_id, predator_rewards)
                    loss_batch += l


                for j in range(prey_num_batches):
                    view_id = prey_view_ids[j]
                    view_values = prey_view_values_list[j]
                    next_view = prey_obs[j*self.args.batch_size:(j+1)*self.args.batch_size]
                    next_view_id, next_view_values = agent_prey.process_view_with_emb_batch(next_view)
                    l = agent_predator.update(view_values, prey_action_batches[j], next_view_values, view_id, prey_rewards)
                    loss_batch += l

                killed = self.env.remove_dead_agents()

                if self.obs_type == 'dense':
                    self.remove_dead_agent_emb(killed)
                else:
                    self.env.remove_dead_agent_emb(killed)

                loss += (loss_batch/(j+1))
                msg = "episode {:03d} episode step {:03d} loss:{:5.4f} predator_reward:{:5.3f} prey_reward:{:5.3f} eps_greedy {:5.3f}".format(episode, i, loss_batch/(j+1), predator_episode_reward/len(predator_obs), prey_episode_reward/len(prey_obs), eps_greedy)
                bar.set_description(msg)
                bar.update(1)


                info = "Episode\t{:03d}\tStep\t{:03d}\tPredator_Reward\t{:5.3f}\tPrey_Reward\t{:5.3f}\tnum_agents\t{:d}\tnum_preys\t{:d}\tnum_predators\t{:d}".format(episode, i, predator_episode_reward/len(predator_obs), prey_episode_reward/len(prey_obs), len(self.env.agents), len(self.env.preys), len(self.env.predators))
                log.write(info+'\n')
                log.flush()
                timesteps += 1

                if self.args.env_type == 'simple_population_dynamics':
                    if i % self.args.increase_every == 0:
                        self.env.increase_prey(self.args.prey_increase_prob)
                        self.env.increase_predator(self.args.predator_increase_prob)
                else:
                    self.env.crossover_prey(self.args.crossover_scope, crossover_rate=self.args.prey_increase_prob)
                    self.env.crossover_predator(self.args.crossover_scope, crossover_rate=self.args.predator_increase_prob)
                if len(self.env.predators) < 2 or len(self.env.preys) < 2 or len(self.env.preys) > 10000 or len(self.env.predators) > 10000:
                    log.close()
                    break

                if i % self.args.update_period:
                    agent_predator.update_params()
                    agent_prey.update_params()


            log_file = os.path.join(log_dir, 'log.txt')
            plot_dynamics(log_file, 0)
            msg = "episode {:03d} avg loss:{:5.4f} avg predator reward:{:5.3f} avg prey reward:{:5.3f} eps_greedy {:5.3f}".format(episode, loss/(i+1), predator_total_reward/(i+1), prey_total_reward/(i+1),eps_greedy)
            bar.set_description(msg)
            bar.update(0)
            bar.refresh()
            bar.close()

            #images = [os.path.join(img_dir, ("{:d}.png".format(j+1))) for j in range(timesteps)]
            #self.env.make_video(images, outvid=os.path.join(img_dir, 'episode_{:d}.avi'.format(rounds)))
            agent_predator.save_model(model_dir, episode, 'model_predator_{:d}.h5'.format(episode))
            agent_prey.save_model(model_dir, episode, 'model_prey_{:d}.h5'.format(episode))

