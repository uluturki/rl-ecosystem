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


class DRQN(nn.Module):
    def __init__(self, args, env, q_net, loss_func, opt, lr=0.001,
                 input_dim=55, hidden_dims=[32, 32],
                 gamma=0.99):
        super(DRQN, self).__init__()
        self.args = args
        self.obs_type = args.obs_type
        self.env = env
        self.agent_emb_dim = args.agent_emb_dim
        self.agent_embeddings = {}
        self.agent_hidden_states = {}
        self.agent_cell_states = {}
        self.agent_target_hidden_states = {}
        self.agent_target_cell_states = {}

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
                view_batches = []
                view_ids = []
                view_values_list = []
                view_agent_embeddings_list = []
                hidden_states = []
                cell_states = []

                trained_env = deepcopy(self.env)

                obs = get_obs(trained_env, only_view=True)
                current_obs = obs
                num_batches = len(current_obs)//self.args.batch_size
                if len(current_obs) > self.args.batch_size*num_batches:
                    num_batches += 1

                next_hidden_states_list = None
                next_cell_states_list = None
                hidden_states_list = None
                cell_states_list = None
                loss_batch = 0
                for j in range(self.args.time_step):
                    actions = []
                    ids = []
                    action_batches = []
                    view_batches = []
                    view_ids = []
                    view_values_list = []
                    view_agent_embeddings_list = []
                    hidden_states = []
                    cell_states = []
                    for k in range(num_batches):
                        view = current_obs[k*self.args.batch_size:(k+1)*self.args.batch_size]

                        batch_id, batch_view, batch_agent_embeddings = self.process_view_with_emb_batch(view)
                        view_agent_embeddings_list.append(batch_agent_embeddings)

                        ## Initial State: Zeros
                        if j == 0:
                            init_hidden_state, init_cell_state = self.q_net.init_hidden_states(len(view))
                            out, hidden_state , cell_state = self.q_net(batch_view, batch_agent_embeddings,
                                                                        Variable(torch.from_numpy(init_hidden_state)).type(self.dtype),
                                                                        Variable(torch.from_numpy(init_cell_state)).type(self.dtype))
                        else:
                            out, hidden_state , cell_state = self.q_net(batch_view, batch_agent_embeddings, hidden_states_list[k], cell_states_list[k])

                        if np.random.rand() < eps_greedy:
                            action = out.max(1)[1].cpu().numpy()
                        else:
                            action = np.random.randint(self.num_actions, size=len(batch_view))

                        ids.extend(batch_id)
                        actions.extend(action)
                        action_batches.append(action)
                        view_batches.append(view)
                        view_ids.append(batch_id)
                        view_values_list.append(batch_view)
                        hidden_states.append(hidden_state)
                        cell_states.append(cell_state)
                    hidden_states_list = hidden_states
                    cell_states_list = cell_states

                    actions = dict(zip(ids, actions))
                    trained_env.take_actions(actions)
                    if self.args.time_step == j+1:
                        next_obs, rewards, killed = get_obs(trained_env)
                    else:
                        next_obs = get_obs(trained_env, only_view=True)


                    next_hidden_states = []
                    next_cell_states = []
                    for k in range(num_batches):
                        view_id = view_ids[k]
                        view_values = view_values_list[k]
                        next_view = next_obs[k*self.args.batch_size:(k+1)*self.args.batch_size]
                        next_view_id, next_view_values, next_agent_embeddings = self.process_view_with_emb_batch(next_view)
                        z, hidden_state, cell_state = self.q_net(view_values, view_agent_embeddings_list[k], hidden_states[k], cell_states[k])

                        ## Init hidden
                        if j == 0:
                            init_next_hidden_state, init_next_cell_state = self.target_q_net.init_hidden_states(len(next_view))
                            next_q_values, next_hidden_state, next_cell_state = self.target_q_net(next_view_values,
                                                                                                  next_agent_embeddings,
                                                                                                  Variable(torch.from_numpy(init_next_hidden_state)).type(self.dtype),
                                                                                                  Variable(torch.from_numpy(init_next_cell_state)).type(self.dtype))
                        else:
                            next_q_values, next_hidden_state, next_cell_state = self.target_q_net(next_view_values, next_agent_embeddings, next_hidden_states_list[k], next_cell_states_list[k])
                        next_hidden_states.append(next_hidden_state)
                        next_cell_states.append(next_cell_state)

                        if self.args.time_step == j+1:
                            q_value = z.gather(1, Variable(torch.Tensor(action_batches[k])).view(len(view_values), 1).type(self.dlongtype))
                            max_next_q_values = next_q_values.max(1)[0].detach()

                            reward_value = []
                            for id in view_id:
                                if id in rewards:
                                    reward_value.append(rewards[id])
                                else:
                                    reward_value.append(0.)
                            reward_value = np.array(reward_value)
                            target = Variable(torch.from_numpy(reward_value)).type(self.dtype) + max_next_q_values * self.gamma
                            target = target.detach().view(len(target), 1) # we do not want to do back-propagation
                            l = self.loss_func(q_value, target)

                            self.opt.zero_grad()

                            l.backward()
                            clip_grad_norm(self.q_net.parameters(), 1.)
                            #torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.)
                            self.opt.step()
                            loss_batch += l.cpu().detach().data.numpy()

                    current_obs = next_obs
                    next_hidden_states_list = next_hidden_states
                    next_cell_states_list = next_cell_states

                actions = []
                ids = []
                action_batches = []
                view_batches = []
                view_ids = []
                view_values_list = []
                view_agent_embeddings_list = []
                hidden_states = []
                cell_states = []
                #obs = get_obs(self.env, only_view=True)
                for k in range(num_batches):
                    view = obs[k*self.args.batch_size:(k+1)*self.args.batch_size]

                    batch_id, batch_view, batch_agent_embeddings, hidden_state, cell_state = self.process_view_with_emb_batch(view, is_states=True)
                    view_agent_embeddings_list.append(batch_agent_embeddings)
                    out, hidden_state , cell_state = self.q_net(batch_view, batch_agent_embeddings, hidden_state, cell_state)
                    hidden_state = hidden_state.detach().cpu().numpy()
                    cell_state = cell_state.detach().cpu().numpy()
                    self.update_states(batch_id, hidden_state,  cell_state)
                    if np.random.rand() < eps_greedy:
                        action = out.max(1)[1].cpu().numpy()
                    else:
                        action = np.random.randint(self.num_actions, size=len(batch_view))

                    ids.extend(batch_id)
                    actions.extend(action)
                    action_batches.append(action)
                    view_batches.append(view)
                    view_ids.append(batch_id)
                    view_values_list.append(batch_view)
                    hidden_states.append(hidden_state)
                    cell_states.append(cell_state)

                actions = dict(zip(ids, actions))
                self.env.take_actions(actions)
                _, rewards, killed = get_obs(self.env)
                episode_reward = np.sum(list(rewards.values()))
                total_reward += episode_reward
                self.env.killed = killed

                increase_predators = self.env.increase_predators
                increase_preys = self.env.increase_preys
                killed = self.env.remove_dead_agents()
                self.remove_dead_agent_emb(killed)

                loss += (loss_batch/(j+1))
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
                elif self.args.env_type != 'simple_population_dynamics_ga_action':
                    if len(self.env.preys) < 5000 and len(self.env.preys) >= 100:
                        self.env.crossover_prey(self.args.crossover_scope, crossover_rate=self.args.prey_increase_prob)
                    if len(self.env.predators) < 5000 and len(self.env.predators)>=100:
                        self.env.crossover_predator(self.args.crossover_scope, crossover_rate=self.args.predator_increase_prob)
                    if len(self.env.preys) < 100:
                        self.env.add_preys(100-len(self.env.preys))
                    if len(self.env.predators) < 100:
                        self.env.add_predators(100-len(self.env.predators))
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
            view_batches = []
            view_ids = []
            view_values_list = []
            view_agent_embeddings_list = []
            hidden_states = []
            cell_states = []
            obs = get_obs(self.env, only_view=True)
            num_batches = len(obs)//self.args.batch_size
            if len(obs) > self.args.batch_size*num_batches:
                num_batches += 1
            for k in range(num_batches):
                view = obs[k*self.args.batch_size:(k+1)*self.args.batch_size]

                batch_id, batch_view, batch_agent_embeddings, hidden_state, cell_state = self.process_view_with_emb_batch(view, is_states=True)
                view_agent_embeddings_list.append(batch_agent_embeddings)
                out, hidden_state , cell_state = self.q_net(batch_view, batch_agent_embeddings, hidden_state, cell_state)

                hidden_state = hidden_state.detach().cpu().numpy()
                cell_state = cell_state.detach().cpu().numpy()
                self.update_states(batch_id, hidden_state,  cell_state)
                action = np.random.randint(self.num_actions, size=len(batch_view))

                ids.extend(batch_id)
                actions.extend(action)
                action_batches.append(action)
                view_batches.append(view)
                view_ids.append(batch_id)
                view_values_list.append(batch_view)
                hidden_states.append(hidden_state)
                cell_states.append(cell_state)

            actions = dict(zip(ids, actions))
            self.env.take_actions(actions)
            _, rewards, killed = get_obs(self.env)
            self.env.killed = killed

            increase_predators = self.env.increase_predators
            increase_preys = self.env.increase_preys
            killed = self.env.remove_dead_agents()
            self.remove_dead_agent_emb(killed)
            total_reward += np.sum(list(rewards.values()))

            killed = self.env.remove_dead_agents()
            self.remove_dead_agent_emb(killed)

            msg = "episode step {:03d}".format(i)
            bar.set_description(msg)
            bar.update(1)

            info = "Step\t{:03d}\tReward\t{:5.3f}\tnum_agents\t{:d}\tnum_preys\t{:d}\tnum_predators\t{:d}\tincrease_predators\t{:d}\tincrease_preys\t{:d}".format(i, total_reward/len(obs), len(self.env.agents), len(self.env.preys), len(self.env.predators), self.env.increase_predators, self.env.increase_preys)
            log.write(info+'\n')
            log.flush()
            timesteps += 1
            killed = self.env.remove_dead_agents()
            self.remove_dead_agent_emb(killed)

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



    def process_view_with_emb_batch(self, input_view, is_states=False):
        batch_embeddings = []
        hidden_states = []
        cell_states = []

        batch_id, batch_view = zip(*input_view)
        for id, view in input_view:
            if id in self.agent_embeddings:
                batch_embeddings.append(self.agent_embeddings[id])
            else:
                new_embedding = np.random.normal(size=[self.agent_emb_dim])
                self.agent_embeddings[id] = new_embedding
                batch_embeddings.append(self.agent_embeddings[id])
#            if target:
#                if id in self.agent_target_hidden_states:
#                    hidden_states.append(self.agent_target_hidden_states[id])
#                    cell_states.append(self.agent_target_cell_states[id])
#                else:
#                    h, c = self.target_q_net.init_hidden_states(1)
#                    self.agent_target_hidden_states[id] = h[0]
#                    self.agent_target_cell_states[id] = c[0]
#                    hidden_states.append(h[0])
#                    cell_states.append(c[0])
#            else:
            if is_states:
                if id in self.agent_hidden_states:
                    hidden_states.append(self.agent_hidden_states[id])
                    cell_states.append(self.agent_cell_states[id])
                else:
                    h, c = self.q_net.init_hidden_states(1)
                    self.agent_hidden_states[id] = h[0]
                    self.agent_cell_states[id] = c[0]
                    hidden_states.append(h[0])
                    cell_states.append(c[0])
        if is_states:
            return batch_id, Variable(torch.from_numpy(np.array(batch_view))).type(self.dtype), Variable(torch.from_numpy(np.array(batch_embeddings))).type(self.dtype), Variable(torch.from_numpy(np.array(hidden_states))).type(self.dtype), Variable(torch.from_numpy(np.array(cell_states))).type(self.dtype)
        else:
            return batch_id, Variable(torch.from_numpy(np.array(batch_view))).type(self.dtype), Variable(torch.from_numpy(np.array(batch_embeddings))).type(self.dtype)

    def remove_dead_agent_emb(self, dead_list):
        for id in dead_list:
            del self.agent_embeddings[id]
            del self.agent_hidden_states[id]
            del self.agent_cell_states[id]

    def update_states(self, ids, hidden_states, cell_states, target=False):
        for i, id in enumerate(ids):
            if target:
                self.agent_target_hidden_states[id] = hidden_states[i]
                self.agent_target_cell_states[id] = cell_states[i]
            else:
                self.agent_hidden_states[id] = hidden_states[i]
                self.agent_cell_states[id] = cell_states[i]

