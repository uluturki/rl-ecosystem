import os, sys

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
from utils import plot_dynamics, plot_diversity
from garl_gym import scenarios


class DQN(nn.Module):
    def __init__(self, args, env, q_net, loss_func, opt, lr=0.001,
                 input_dim=55, hidden_dims=[32, 32],
                 gamma=0.99):
        super(DQN, self).__init__()
        self.args = args
        self.obs_type = args.obs_type
        self.env = env
        self.agent_emb_dim = args.agent_emb_dim
        self.agent_embeddings = {}

        self.num_actions = args.num_actions
        self.loss_func = loss_func
        self.video_flag= args.video_flag

        self.gamma = gamma

        if torch.cuda.is_available():
            self.dtype = torch.cuda.FloatTensor
            self.dlongtype = torch.cuda.LongTensor
        else:
            self.dtype = torch.FloatTensor
            self.dlongtype = torch.LongTensor

        self.q_net = q_net.type(self.dtype)
        self.opt = opt(self.q_net.parameters(), lr)
        self.target_q_net = deepcopy(q_net).type(self.dtype)

    def train(self,
              episodes=100,
              episode_step=500,
              random_step=5000,
              min_greedy=0.3,
              max_greedy=0.9,
              greedy_step=6000,
              update_period=10):

        if self.args.env_type == 'simple_population_dynamics':
            get_obs = scenarios.simple_population_dynamics.get_obs
        elif self.args.env_type == 'simple_population_dynamics_ga':
            get_obs = scenarios.simple_population_dynamics_ga.get_obs
        elif self.args.env_type == 'simple_population_dynamics_ga_utility':
            get_obs = scenarios.simple_population_dynamics_ga_utility.get_obs
        elif self.args.env_type == 'simple_population_dynamics_ga_action':
            get_obs = scenarios.simple_population_dynamics_ga_action.get_obs
            assert self.num_actions == 5, 'Expected the number of actions 5, but got {:d}'.format(self.num_actions)

        eps_greedy = min_greedy
        g_step = (max_greedy - min_greedy) / greedy_step
        #g_step = 0
        #eps_greedy = 0.9

        rounds = 0
        model_dir = os.path.join('results', self.args.env_type, 'exp_{:d}'.format(self.args.experiment_id), 'models')
        try:
            os.makedirs(model_dir)
        except:
            shutil.rmtree(model_dir)
            os.makedirs(model_dir)

        for episode in range(episodes):
            loss = 0
            total_reward = 0
            bar = tqdm()


            #if episode==0 or len(self.env.predators) < 2 or len(self.env.preys) < 2 or len(self.env.preys) > 15000 or len(self.env.predators) > 15000:
            if episode==0:
                #or len(self.env.predators) < 2 or len(self.env.preys) < 2 or len(self.env.preys) > 15000 or len(self.env.predators) > 15000:
                obs = self.env.reset()

                img_dir = os.path.join('results', self.args.env_type, 'exp_{:d}'.format(self.args.experiment_id), 'images',str(rounds))
                log_dir = os.path.join('results', self.args.env_type, 'exp_{:d}'.format(self.args.experiment_id), 'logs', str(rounds))
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

            for i in range(episode_step):
                episode_reward = 0
                if self.video_flag:
                    self.env.dump_image(os.path.join(img_dir, '{:d}.png'.format(timesteps+1)))

                eps_greedy += g_step
                eps_greedy = np.clip(eps_greedy, min_greedy, max_greedy)


                actions = []
                ids = []
                action_batches = []
                #obs = self.env.render(only_view=True)
                obs = get_obs(self.env, only_view=True)
                view_batches = []
                view_ids = []
                view_values_list = []
                view_agent_embeddings_list = []


                for j in range(len(obs)//self.args.batch_size+1):
                    view = obs[j*self.args.batch_size:(j+1)*self.args.batch_size]
                    if len(view) == 0:
                        continue

                    if self.obs_type == 'conv_with_id':
                        batch_id, batch_view, batch_agent_embeddings = self.process_view_with_emb_batch(view)
                        view_agent_embeddings_list.append(batch_agent_embeddings)
                        if np.random.rand() < eps_greedy:
                            action = self.q_net(batch_view, batch_agent_embeddings).max(1)[1].cpu().numpy()
                        else:
                            action = np.random.randint(self.num_actions, size=len(batch_view))
                    else:
                        batch_id, batch_view = self.process_view_with_emb_batch(view)
                        if np.random.rand() < eps_greedy:
                            action = self.q_net(batch_view).max(1)[1].cpu().numpy()
                        else:
                            action = np.random.randint(self.num_actions, size=len(batch_view))

                    ids.extend(batch_id)
                    actions.extend(action)
                    action_batches.append(action)
                    view_batches.append(view)
                    view_ids.append(batch_id)
                    view_values_list.append(batch_view)
                num_batches = j

                actions = dict(zip(ids, actions))
                #next_view_batches, rewards = self.env.step(actions)

                self.env.take_actions(actions)
                next_view_batches, rewards, killed = get_obs(self.env)
                self.env.killed = killed
                episode_reward += np.sum(list(rewards.values()))
                total_reward += (episode_reward / len(obs))

                loss_batch = 0
                for j in range(num_batches+1):
                    #view_id, view_values = self.process_view_with_emb_batch(view)
                    view_id = view_ids[j]
                    view_values = view_values_list[j]
                    next_view = obs[j*self.args.batch_size:(j+1)*self.args.batch_size]
                    if self.obs_type == 'conv_with_id':
                        next_view_id, next_view_values, next_agent_embeddings = self.process_view_with_emb_batch(next_view)
                        z = self.q_net(view_values, view_agent_embeddings_list[j])
                    else:
                        next_view_id, next_view_values = self.process_view_with_emb_batch(next_view)
                        z = self.q_net(view_values)
                    z = z.gather(1, Variable(torch.Tensor(action_batches[j])).view(len(view_values), 1).type(self.dlongtype))

                    if self.obs_type == 'conv_with_id':
                        next_q_values = self.target_q_net(next_view_values, next_agent_embeddings).max(1)[0].detach()
                    else:
                        next_q_values = self.target_q_net(next_view_values).max(1)[0].detach()

                    reward_value = []
                    for id in view_id:
                        if id in rewards:
                            reward_value.append(rewards[id])
                        else:
                            reward_value.append(0.)

                    reward_value = np.array(reward_value)
                    target = Variable(torch.from_numpy(reward_value)).type(self.dtype) + next_q_values * self.gamma
                    target = target.detach().view(len(target), 1) # we do not want to do back-propagation

                    l = self.loss_func(z, target)

                    self.opt.zero_grad()

                    l.backward()
                    clip_grad_norm(self.q_net.parameters(), 1.)
                    #torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.)
                    self.opt.step()
                    loss_batch += l.cpu().detach().data.numpy()

                increase_predators = self.env.increase_predators
                increase_preys = self.env.increase_preys
                killed = self.env.remove_dead_agents()
                if self.obs_type == 'dense' or self.obs_type == 'conv_with_id':
                    self.remove_dead_agent_emb(killed)
                else:
                    self.env.remove_dead_agent_emb(killed)

                loss += (loss_batch/(j+1))
                view_batches = next_view_batches
                msg = "episode {:03d} episode step {:03d} loss:{:5.4f} reward:{:5.3f} eps_greedy {:5.3f}".format(episode, i, loss_batch/(j+1), episode_reward/len(obs), eps_greedy)
                bar.set_description(msg)
                bar.update(1)

                info = "Step\t{:03d}\tReward\t{:5.3f}\tnum_agents\t{:d}\tnum_preys\t{:d}\tnum_predators\t{:d}\tincrease_predators\t{:d}\tincrease_preys\t{:d}\tkilled_agents\t{:d}".format(i, episode_reward/len(obs), len(self.env.agents), len(self.env.preys), len(self.env.predators),increase_predators, increase_preys, len(killed))
                log.write(info+'\n')
                log.flush()
                timesteps += 1

                if self.args.env_type == 'simple_population_dynamics':
                    if i % self.args.increase_every == 0:
                        self.env.increase_prey(self.args.prey_increase_prob)
                        self.env.increase_predator(self.args.predator_increase_prob)
                elif self.args.env_type in ['simple_population_dynamics_ga', 'simple_population_dynamics_ga_utility']:
                    self.env.crossover_prey(self.args.crossover_scope, crossover_rate=self.args.prey_increase_prob)
                    #self.env.increase_prey(self.args.prey_increase_prob)
                    self.env.crossover_predator(self.args.crossover_scope, crossover_rate=self.args.predator_increase_prob)
                elif self.args.env_type != 'simple_population_dynamics_ga_action':
                    if len(self.env.preys) < 5000 and len(self.env.preys) >= 100:
                        self.env.crossover_prey(self.args.crossover_scope, crossover_rate=self.args.prey_increase_prob)
                    if len(self.env.predators) < 5000 and len(self.env.predators)>=100:
                        self.env.crossover_predator(self.args.crossover_scope, crossover_rate=self.args.predator_increase_prob)
                    if len(self.env.preys) < 100:
                        self.env.add_preys(100-len(self.env.preys))
                    if len(self.env.predators) < 100:
                        self.env.add_predators(100-len(self.env.predators))
                if len(self.env.predators) < 2 or len(self.env.preys) < 2 or len(self.env.preys) > 10000 or len(self.env.predators) > 10000:
                    log.close()
                    break
                #if len(self.env.predators) < 2 or len(self.env.preys) < 2 or len(self.env.preys) > 15000 or len(self.env.predators) > 15000:
                #    log.close()
                #    break

                if i % update_period:
                    self.update_params()


            log_file = os.path.join(log_dir, 'log.txt')
            plot_dynamics(log_file, 0)
            msg = "episode {:03d} avg loss:{:5.4f} avg reward:{:5.3f} eps_greedy {:5.3f}".format(episode, loss/(i+1), total_reward/(i+1), eps_greedy)
            bar.set_description(msg)
            bar.update(0)
            bar.refresh()
            bar.close()

            #images = [os.path.join(img_dir, ("{:d}.png".format(j+1))) for j in range(timesteps)]
            #self.env.make_video(images, outvid=os.path.join(img_dir, 'episode_{:d}.avi'.format(rounds)))
            self.save_model(model_dir, episode)

    def update(self, view_values, action_batches, next_view_values, view_id, rewards):
        z = self.q_net(view_values)
        z = z.gather(1, Variable(torch.Tensor(action_batches)).view(len(view_values), 1).type(self.dlongtype))

        next_q_values = self.target_q_net(next_view_values).max(1)[0].detach()

        reward_value = []
        for id in view_id:
            if id in rewards:
                reward_value.append(rewards[id])
            else:
                reward_value.append(0.)

        reward_value = np.array(reward_value)
        target = Variable(torch.from_numpy(reward_value)).type(self.dtype) + next_q_values * self.gamma
        target = target.detach().view(len(target), 1) # we do not want to do back-propagation

        l = self.loss_func(z, target)

        self.opt.zero_grad()

        l.backward()
        #clip_grad_norm(self.q_net.parameters(), 1.)
        #torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.)
        self.opt.step()
        return l.cpu().detach().data.numpy()

    def take_action(self, batch_view):
        return self.q_net(batch_view).max(1)[1].cpu().numpy()

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

        total_reward = 0
        bar = tqdm()


        img_dir = os.path.join('results', self.args.env_type, 'exp_{:d}'.format(self.args.experiment_id), 'test_images', str(self.args.test_id))
        log_dir = os.path.join('results', self.args.env_type, 'exp_{:d}'.format(self.args.experiment_id), 'test_logs', str(self.args.test_id))

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

        timesteps = 0

        for i in range(test_step):
            if self.video_flag:
                self.env.dump_image(os.path.join(img_dir, '{:d}.png'.format(timesteps+1)))

            if self.args.env_type == 'simple_population_dynamics_ga_utility' and i % 200 == 0:
                plot_diversity(self.env.predators.values(), self.env.preys.values(), log_dir, i)


            actions = []
            ids = []
            action_batches = []
            #obs = self.env.render(only_view=True)
            obs = get_obs(self.env, only_view=True)
            view_batches = []
            view_ids = []
            view_values_list = []
            view_agent_embeddings_list = []

            for j in range(len(obs)//self.args.batch_size+1):
                view = obs[j*self.args.batch_size:(j+1)*self.args.batch_size]
                if len(view) == 0:
                    continue
                if self.obs_type == 'conv_with_id':
                    batch_id, batch_vbiew, batch_agent_embeddings = self.process_view_with_emb_batch(view)
                    action = self.q_net(batch_view, batch_agent_embeddings).max(1)[1].cpu().numpy()
                    view_agent_embeddings_list.append(batch_agent_embeddings)
                else:
                    batch_id, batch_view = self.process_view_with_emb_batch(view)
                    action = self.q_net(batch_view).max(1)[1].cpu().numpy()

                print(action)
                ids.extend(batch_id)
                actions.extend(action)
                action_batches.append(action)
                view_batches.append(view)
                view_ids.append(batch_id)
                view_values_list.append(batch_view)


            actions = dict(zip(ids, actions))
            #next_view_batches, rewards = self.env.step(actions)
            self.env.take_actions(actions)
            next_view_batches, rewards, killed = get_obs(self.env)
            self.env.killed = killed
            total_reward += np.sum(list(rewards.values()))

            killed = self.env.remove_dead_agents()
            if self.obs_type == 'dense':
                self.remove_dead_agent_emb(killed)
            else:
                self.env.remove_dead_agent_emb(killed)

            msg = "episode step {:03d}".format(i)
            bar.set_description(msg)
            bar.update(1)

            info = "Step\t{:03d}\tReward\t{:5.3f}\tnum_agents\t{:d}\tnum_preys\t{:d}\tnum_predators\t{:d}\tincrease_predators\t{:d}\tincrease_preys\t{:d}".format(i, total_reward/len(obs), len(self.env.agents), len(self.env.preys), len(self.env.predators), self.env.increase_predators, self.env.increase_preys)
            log.write(info+'\n')
            log.flush()
            timesteps += 1
            killed = self.env.remove_dead_agents()
            if self.obs_type == 'dense':
                self.remove_dead_agent_emb(killed)
            else:
                self.env.remove_dead_agent_emb(killed)

            if self.args.env_type == 'simple_population_dynamics':
                if i % self.args.increase_every == 0:
                    self.env.increase_prey(self.args.prey_increase_prob)
                    self.env.increase_predator(self.args.predator_increase_prob)
            elif self.args.env_type != 'simple_population_dynamics_ga_action':
                self.env.crossover_prey(self.args.crossover_scope, crossover_rate=self.args.prey_increase_prob)
                self.env.crossover_predator(self.args.crossover_scope, crossover_rate=self.args.predator_increase_prob)


            if len(self.env.predators) < 1 or len(self.env.preys) < 1 or len(self.env.predators) > 20000 or len(self.env.preys) > 20000:
                log.close()
                break
        #images = [os.path.join(img_dir, ("{:d}.png".format(j+1))) for j in range(timesteps)]
        #self.env.make_video(images, outvid=os.path.join(img_dir, 'episode_{:d}.avi')


    def save_model(self, model_dir, episode, file_name=None):
        if file_name is None:
            torch.save(self.q_net, os.path.join(model_dir, "model_{:d}.h5".format(episode)))
        else:
            torch.save(self.q_net, os.path.join(model_dir, file_name))



    def update_params(self):
        self.target_q_net = deepcopy(self.q_net)

    def take_action(self, state):
        raise NotImplementedError



    def process_view_with_emb_batch(self, input_view):
        batch_id = []
        batch_view = []
        batch_embeddings = []

        if self.obs_type == 'conv':
            batch_id, batch_view = zip(*input_view)
        else:
            if self.obs_type == 'conv_with_id':
                batch_id, batch_view = zip(*input_view)
            for id, view in input_view:
                if self.obs_type == 'dense':
                    if id in self.agent_embeddings:
                        new_view = np.concatenate((self.agent_embeddings[id], view), 0)
                        batch_view.append(new_view)
                        batch_id.append(id)
                    else:
                        new_embedding = np.random.normal(size=[self.agent_emb_dim])
                        self.agent_embeddings[id] = new_embedding
                        new_view = np.concatenate((new_embedding, view), 0)
                        batch_view.append(new_view)
                        batch_id.append(id)
                else:
                    if id in self.agent_embeddings:
                        batch_embeddings.append(self.agent_embeddings[id])
                    else:
                        new_embedding = np.random.normal(size=[self.agent_emb_dim])
                        self.agent_embeddings[id] = new_embedding
                        batch_embeddings.append(self.agent_embeddings[id])
        if self.obs_type == 'conv_with_id':
            return batch_id, Variable(torch.from_numpy(np.array(batch_view))).type(self.dtype), Variable(torch.from_numpy(np.array(batch_embeddings))).type(self.dtype)
        return batch_id, Variable(torch.from_numpy(np.array(batch_view))).type(self.dtype)

    def remove_dead_agent_emb(self, dead_list):
        for id in dead_list:
            del self.agent_embeddings[id]


