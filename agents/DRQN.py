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
#from torch.utils.tensorboard import SummaryWriter


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

        if hasattr(args, 'experiment_type'):
            self.experiment_type = args.experiment_type
        else:
            self.experiment_type = None


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
        elif self.args.env_type == 'complex_population_dynamics':
            get_obs = scenarios.complex_population_dynamics.get_obs
        elif self.args.env_type == 'genetic_population_dynamics':
            get_obs = scenarios.genetic_population_dynamics.get_obs

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
            if episode==0 or len(self.env.predators) < 2 or len(self.env.preys) < 2 or len(self.env.preys) > self.args.prey_capacity or len(self.env.predators) > self.args.predator_capacity:
            #if episode==0:
                #or len(self.env.predators) < 2 or len(self.env.preys) < 2 or len(self.env.preys) > 15000 or len(self.env.predators) > 15000:
                obs = self.env.reset()
                img_dir, log_dir = self.create_dir(rounds)
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
                    outs = []
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

                        tmp_action = []
                        for m in range(len(view)):
                            if np.random.rand() < eps_greedy:
                                tmp_action.append(int(out[m].max(0)[1].cpu()))
                            else:
                                tmp_action.append(np.random.randint(self.num_actions))

                        ids.extend(batch_id)
                        actions.extend(tmp_action)
                        action_batches.append(tmp_action)
                        view_batches.append(view)
                        view_ids.append(batch_id)
                        view_values_list.append(batch_view)
                        hidden_states.append(hidden_state)
                        cell_states.append(cell_state)
                        outs.append(out)
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
                        #z = outs[k]

                        ## Init hidden
                        if j == 0:
                            init_next_hidden_state, init_next_cell_state = self.target_q_net.init_hidden_states(len(next_view))
                            next_q_values, next_hidden_state, next_cell_state = self.target_q_net(next_view_values,
                                                                                                  next_agent_embeddings,
                                                                                                  Variable(torch.from_numpy(init_next_hidden_state)).type(self.dtype),
                                                                                                  Variable(torch.from_numpy(init_next_cell_state)).type(self.dtype))
                        else:
                            next_q_values, next_hidden_state, next_cell_state = self.target_q_net(next_view_values, next_agent_embeddings, next_hidden_states_list[k], next_cell_states_list[k])
                        next_hidden_states.append(next_hidden_state.detach())
                        next_cell_states.append(next_cell_state.detach())

                        if self.args.time_step == j+1:
                            prior_action = self.q_net(next_view_values, next_agent_embeddings, next_hidden_states_list[k], next_cell_states_list[k])[0].max(1)[1].detach()
                            q_value = z.gather(1, Variable(torch.Tensor(action_batches[k])).view(len(view_values), 1).type(self.dlongtype))
                            #max_next_q_values = next_q_values.max(1)[0].detach()
                            max_next_q_values = next_q_values.gather(1, prior_action.view(-1, 1).type(self.dlongtype)).detach()

                            reward_value = []
                            for m, id in enumerate(view_id):
                                if id in rewards.keys():
                                    reward_value.append(rewards[id])
                                    #if rewards[id] < -0.001:
                                        #max_next_q_values[m] = 0
                                else:
                                    reward_value.append(0.)
                            reward_value = np.array(reward_value)
                            target = Variable(torch.from_numpy(reward_value)).type(self.dtype).view(-1, 1) + max_next_q_values * self.gamma
                            #target = target.detach().view(len(target), 1) # we do not want to do back-propagation
                            target = target.detach()
                            l = self.loss_func(q_value, target)

                            self.opt.zero_grad()

                            l.backward()
                            clip_grad_norm(self.q_net.parameters(), 0.1)
                            #torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.)
                            self.opt.step()
                            loss_batch += l.cpu().detach().data.numpy()



                    current_obs = next_obs
                    next_hidden_states_list = next_hidden_states
                    next_cell_states_list = next_cell_states

                actions = []
                ids = []
                #obs = get_obs(self.env, only_view=True)
                for k in range(num_batches):
                    with torch.no_grad():
                        view = obs[k*self.args.batch_size:(k+1)*self.args.batch_size]

                        batch_id, batch_view, batch_agent_embeddings, hidden_state, cell_state = self.process_view_with_emb_batch(view, is_states=True)
                        view_agent_embeddings_list.append(batch_agent_embeddings)
                        out, hidden_state , cell_state = self.q_net(batch_view, batch_agent_embeddings, hidden_state, cell_state)
                        hidden_state = hidden_state.detach().cpu().numpy()
                        cell_state = cell_state.detach().cpu().numpy()
                        self.update_states(batch_id, hidden_state,  cell_state)
                        tmp_action = []
                        for m in range(len(view)):
                            if np.random.rand() < eps_greedy:
                                tmp_action.append(int(out[m].max(0)[1].cpu()))
                            else:
                                tmp_action.append(np.random.randint(self.num_actions))


                        ids.extend(batch_id)
                        actions.extend(tmp_action)

                actions = dict(zip(ids, actions))
                self.env.take_actions(actions)
                _, rewards, killed = get_obs(self.env)
                episode_reward = np.sum(list(rewards.values()))
                total_reward += episode_reward
                self.env.killed = killed


                #predator_rewards, prey_rewards = self.env.return_total_reward(rewards)
                #writer.add_scalar('predator_reward', predator_rewards, timesteps)
                #writer.add_scalar('prey_reward', prey_rewards, timesteps)

                increase_predators = self.env.increase_predators
                increase_preys = self.env.increase_preys
                killed = self.env.remove_dead_agents()
                self.remove_dead_agent_emb(killed)

                #if i % 4 == 0:
                #    self.reset_states()

                loss += (loss_batch/(num_batches))
                msg = "episode {:03d} episode step {:03d} loss:{:5.4f} reward:{:5.3f} eps_greedy {:5.3f}".format(episode, i, loss_batch/num_batches, episode_reward/len(obs), eps_greedy)
                bar.set_description(msg)
                bar.update(1)


                timesteps += 1
                if self.args.env_type == 'simple_population_dynamics':
                    info = "Step\t{:03d}\tReward\t{:5.3f}\tnum_agents\t{:d}\tnum_preys\t{:d}\tnum_predators\t{:d}\tincrease_predators\t{:d}\tincrease_preys\t{:d}\tkilled_agents\t{:d}".format(i, episode_reward/len(obs), len(self.env.agents), len(self.env.preys), len(self.env.predators),increase_predators, increase_preys, len(killed))
                    log.write(info+'\n')
                    log.flush()
                    if i % self.args.increase_every == 0:
                        self.env.increase_prey(self.args.prey_increase_prob)
                        self.env.increase_predator(self.args.predator_increase_prob)
                elif self.args.env_type == 'complex_population_dynamics':
                    info = "Step\t{:03d}\tReward\t{:5.3f}\tnum_agents\t{:d}\tnum_preys\t{:d}\tnum_predators\t{:d}\tincrease_predators\t{:d}\tincrease_preys\t{:d}\tkilled_agents\t{:d}\taverage_predator_health\t{:5.3f}\taverage_prey_health\t{:5.3f}\taverage_predator_attack\t{:5.3f}\taverage_prey_attack\t{:5.3f}\taverage_predator_resilience\t{:5.3f}\taverage_prey_resilience\t{:5.3f}\taverage_predator_speed\t{:5.3f}\taverage_prey_speed\t{:5.3f}\tvar_predator_health\t{:5.3f}\tvar_prey_health\t{:5.3f}\tvar_predator_attack\t{:5.3f}\tvar_prey_attack\t{:5.3f}\tvar_predator_resilience\t{:5.3f}\var_prey_resilience\t{:5.3f}\tvar_predator_speed\t{:5.3f}\tvar_prey_speed\t{:5.3f}".format(i, episode_reward/len(obs), len(self.env.agents), len(self.env.preys), len(self.env.predators),increase_predators, increase_preys, len(killed), np.mean(self.env.predator_health), np.mean(self.env.prey_health), np.mean(self.env.predator_attack), np.mean(self.env.prey_attack), np.mean(self.env.predator_resilience), np.mean(self.env.prey_resilience), np.mean(self.env.predator_speed), np.mean(self.env.prey_speed), np.var(self.env.predator_health), np.var(self.env.prey_health), np.var(self.env.predator_attack), np.var(self.env.prey_attack), np.var(self.env.predator_resilience), np.var(self.env.prey_resilience), np.var(self.env.predator_speed), np.var(self.env.prey_speed))
                    log.write(info+'\n')
                    log.flush()
                    self.env.crossover_prey(self.args.crossover_scope, crossover_rate=self.args.prey_increase_prob, mutation_prob=self.args.mutation_prob)
                    self.env.crossover_predator(self.args.crossover_scope, crossover_rate=self.args.predator_increase_prob, mutation_prob=self.args.mutation_prob)
                    if i % self.args.increase_every == 0:
                        self.env.add_preys(1)
                        self.env.add_predators(1)
                elif self.args.env_type == 'genetic_population_dynamics':
                    info = "Step\t{:03d}\tReward\t{:5.3f}\tnum_agents\t{:d}\tnum_preys\t{:d}\tnum_predators\t{:d}\tincrease_predators\t{:d}\tincrease_preys\t{:d}\tkilled_agents\t{:d}\taverage_predator_health\t{:5.3f}\taverage_prey_health\t{:5.3f}\taverage_predator_attack\t{:5.3f}\taverage_prey_attack\t{:5.3f}\taverage_predator_resilience\t{:5.3f}\taverage_prey_resilience\t{:5.3f}\taverage_predator_speed\t{:5.3f}\taverage_prey_speed\t{:5.3f}\tvar_predator_health\t{:5.3f}\tvar_prey_health\t{:5.3f}\tvar_predator_attack\t{:5.3f}\tvar_prey_attack\t{:5.3f}\tvar_predator_resilience\t{:5.3f}\var_prey_resilience\t{:5.3f}\tvar_predator_speed\t{:5.3f}\tvar_prey_speed\t{:5.3f}".format(i, episode_reward/len(obs), len(self.env.agents), len(self.env.preys), len(self.env.predators),increase_predators, increase_preys, len(killed), np.mean(self.env.predator_health), np.mean(self.env.prey_health), np.mean(self.env.predator_attack), np.mean(self.env.prey_attack), np.mean(self.env.predator_resilience), np.mean(self.env.prey_resilience), np.mean(self.env.predator_speed), np.mean(self.env.prey_speed), np.var(self.env.predator_health), np.var(self.env.prey_health), np.var(self.env.predator_attack), np.var(self.env.prey_attack), np.var(self.env.predator_resilience), np.var(self.env.prey_resilience), np.var(self.env.predator_speed), np.var(self.env.prey_speed))
                    log.write(info+'\n')
                    log.flush()
                    self.env.crossover_prey(self.args.prey_increase_prob, mutation_prob=self.args.mutation_prob)
                    self.env.crossover_predator(self.args.predator_increase_prob, mutation_prob=self.args.mutation_prob)
                elif self.args.env_type != 'simple_population_dynamics_ga_action':
                    info = "Step\t{:03d}\tReward\t{:5.3f}\tnum_agents\t{:d}\tnum_preys\t{:d}\tnum_predators\t{:d}\tincrease_predators\t{:d}\tincrease_preys\t{:d}\tkilled_agents\t{:d}".format(i, episode_reward/len(obs), len(self.env.agents), len(self.env.preys), len(self.env.predators),increase_predators, increase_preys, len(killed))
                    log.write(info+'\n')
                    log.flush()
                    self.env.crossover_prey(self.args.crossover_scope, crossover_rate=self.args.prey_increase_prob)
                    self.env.crossover_predator(self.args.crossover_scope, crossover_rate=self.args.predator_increase_prob)
                    self.env.add_preys(1)
                    self.env.add_predators(1)
                if len(self.env.predators) < 2 or len(self.env.preys) < 2 or len(self.env.preys) > self.args.prey_capacity or len(self.env.predators) > self.args.predator_capacity:
                    log.close()
                    break

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
       # try:
       #     os.makedirs(log_dir)
       # except:
       #     shutil.rmtree(log_dir)
       #     os.makedirs(log_dir)
        log = open(os.path.join(log_dir, 'log.txt'), 'w')
        log_local = open(os.path.join(log_dir, 'log_division.txt'), 'w')

        timesteps = 0

        for i in range(test_step):
            if self.video_flag:
                self.env.dump_image(os.path.join(img_dir, '{:d}.png'.format(timesteps+1)))

            if self.args.env_type == 'simple_population_dynamics_ga_utility' and i % 200 == 0:
                plot_diversity(self.env.predators.values(), self.env.preys.values(), log_dir, i)

            if i % 100 == 0:
                plot_dynamics(os.path.join(log_dir, 'log.txt'), st=0)


            actions = []
            ids = []
            action_batches = []
            view_batches = []
            view_ids = []
            view_values_list = []
            view_agent_embeddings_list = []
            obs = get_obs(self.env, only_view=True)
            num_batches = len(obs)//self.args.batch_size
            if len(obs) > self.args.batch_size*num_batches:
                num_batches += 1
            for k in range(num_batches):
                with torch.no_grad():
                    view = obs[k*self.args.batch_size:(k+1)*self.args.batch_size]

                    batch_id, batch_view, batch_agent_embeddings, hidden_state, cell_state = self.process_view_with_emb_batch(view, is_states=True)
                    view_agent_embeddings_list.append(batch_agent_embeddings)
                    out, hidden_state , cell_state = self.q_net(batch_view, batch_agent_embeddings, hidden_state, cell_state)

                    hidden_state = hidden_state.detach().cpu().numpy()
                    cell_state = cell_state.detach().cpu().numpy()
                    self.update_states(batch_id, hidden_state,  cell_state)
                    #action = out.max(1)[1].cpu().numpy()
                    tmp_action = []
                    for m in range(len(view)):
                        if np.random.rand() < 0.95:
                            tmp_action.append(int(out[m].max(0)[1].cpu()))
                        else:
                            tmp_action.append(np.random.randint(self.num_actions))

                    ids.extend(batch_id)
                    actions.extend(tmp_action)
                    action_batches.append(tmp_action)
                    view_batches.append(view)
                    view_ids.append(batch_id)
                    view_values_list.append(batch_view)

            actions = dict(zip(ids, actions))
            self.env.take_actions(actions)
            _, rewards, killed = get_obs(self.env)
            episode_reward = np.sum(list(rewards.values()))
            total_reward += episode_reward
            self.env.killed = killed

            predator_rewards, prey_rewards = self.env.return_total_reward(rewards)
            #writer.add_scalar('predator_reward', predator_rewards, timesteps)
                #writer.add_scalar('prey_reward', prey_rewards, timesteps)

            increase_predators = self.env.increase_predators
            increase_preys = self.env.increase_preys
            killed = self.env.remove_dead_agents()
            self.remove_dead_agent_emb(killed)
            total_reward += np.sum(list(rewards.values()))
            #if i % 4 == 0:
            #    self.reset_states()



            msg = "episode step {:03d}".format(i)
            bar.set_description(msg)
            bar.update(1)

            info = "Step\t{:03d}\tReward\t{:5.3f}\tnum_agents\t{:d}\tnum_preys\t{:d}\tnum_predators\t{:d}\tincrease_predators\t{:d}\tincrease_preys\t{:d}".format(i, total_reward/len(obs), len(self.env.agents), len(self.env.preys), len(self.env.predators), self.env.increase_predators, self.env.increase_preys)
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
                new_embedding = np.random.normal(size=[self.agent_emb_dim-1])
                if self.env.agents[id].predator:
                    new_embedding = np.concatenate([new_embedding, [1.]])
                else:
                    new_embedding = np.concatenate([new_embedding, [0.]])
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
                if id in self.agent_hidden_states.keys():
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


    def one_iteration(self, timestep):
        total_reward = 0

        actions = []
        ids = []
        action_batches = []
        view_batches = []
        view_ids = []
        view_values_list = []
        view_agent_embeddings_list = []
        hidden_states = []
        cell_states = []
        obs = self.get_obs(self.env, only_view=True)
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
            action = out.max(1)[1].cpu().numpy()

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
        _, rewards, killed = self.get_obs(self.env)
        episode_reward = np.sum(list(rewards.values()))
        total_reward += episode_reward
        self.env.killed = killed

        predator_rewards, prey_rewards = self.env.return_total_reward(rewards)
        #writer.add_scalar('predator_reward', predator_rewards, timesteps)
            #writer.add_scalar('prey_reward', prey_rewards, timesteps)

        increase_predators = self.env.increase_predators
        increase_preys = self.env.increase_preys
        killed = self.env.remove_dead_agents()
        self.remove_dead_agent_emb(killed)
        total_reward += np.sum(list(rewards.values()))

        if self.args.env_type == 'simple_population_dynamics':
            if timestep % self.args.increase_every == 0:
                self.env.increase_prey(self.args.prey_increase_prob)
                self.env.increase_predator(self.args.predator_increase_prob)
        elif self.args.env_type == 'complex_population_dynamics':
            timesteps += 1
            self.env.crossover_prey(self.args.crossover_scope, crossover_rate=self.args.prey_increase_prob, mutation_prob=self.args.mutation_prob)
            self.env.crossover_predator(self.args.crossover_scope, crossover_rate=self.args.predator_increase_prob, mutation_prob=self.args.mutation_prob)
            self.env.add_preys(1)
            self.env.add_predators(1)
        elif self.args.env_type != 'simple_population_dynamics_ga_action':
            self.env.crossover_prey(self.args.crossover_scope, crossover_rate=self.args.prey_increase_prob)
            self.env.crossover_predator(self.args.crossover_scope, crossover_rate=self.args.predator_increase_prob)
            self.env.add_preys(1)
            #if len(self.env.predators) < 200:
            self.env.add_predators(1)


    def remove_dead_agent_emb(self, dead_list):
        for id in dead_list:
            if id in self.agent_embeddings:
                del self.agent_embeddings[id]
                del self.agent_hidden_states[id]
                del self.agent_cell_states[id]

    def update_states(self, ids, hidden_states, cell_states, target=False):
        for i, id in enumerate(ids):
            self.agent_hidden_states[id] = hidden_states[i]
            self.agent_cell_states[id] = cell_states[i]

    def reset_states(self):
        self.agent_hidden_states = {}
        self.agent_cell_states = {}

    def get_states(self, ids):
        hidden_states = []
        cell_states = []
        for i, id in enumerate(ids):
            if id in self.agent_hidden_states:
                hidden_states.append(self.agent_hidden_states[id])
                cell_states.append(self.agent_cell_states[id])
            else:
                hidden_state, cell_state = self.q_net.init_hidden_states(1)
                hidden_states.append(hidden_state[0])
                cell_states.append(cell_state[0])

        hidden_states = np.array(hidden_states)
        cell_states = np.array(cell_states)
        return torch.from_numpy(hidden_states).type(self.dtype), torch.from_numpy(cell_states).type(self.dtype)



    def train_with_variation(self,
              episodes=100,
              episode_step=500,
              random_step=5000,
              min_greedy=0.3,
              max_greedy=0.9,
              greedy_step=6000,
              update_period=10):

        if self.args.env_type == 'simple_population_dynamics':
            get_obs = scenarios.simple_population_dynamics.get_obs_with_variation
        elif self.args.env_type == 'simple_population_dynamics_ga':
            get_obs = scenarios.simple_population_dynamics_ga.get_obs_with_variation
        elif self.args.env_type == 'complex_population_dynamics':
            get_obs = scenarios.complex_population_dynamics.get_obs

        eps_greedy = 0.95

        rounds = 0

        self.trained_q_net = deepcopy(self.q_net)

        for episode in range(episodes):
            loss = 0
            total_reward = 0
            bar = tqdm()
            #if episode==0 or len(self.env.predators) < 2 or len(self.env.preys) < 2 or len(self.env.preys) > 15000 or len(self.env.predators) > 15000:
            if episode==0 or len(self.env.predator_agents) < 2 or len(self.env.prey_agents) < 2 or len(self.env.prey_agents) > self.args.prey_capacity or len(self.env.predator_agents) > self.args.predator_capacity:
            #if episode==0:
                #or len(self.env.predators) < 2 or len(self.env.preys) < 2 or len(self.env.preys) > 15000 or len(self.env.predators) > 15000:
                obs = self.env.reset()
                img_dir, log_dir = self.create_dir(rounds)
                log = open(os.path.join(log_dir, 'log.txt'), 'w')
                rounds += 1
                timesteps = 0

            for i in range(episode_step):
                episode_reward = 0
                if self.video_flag:
                    self.env.dump_image(os.path.join(img_dir, '{:d}.png'.format(timesteps+1)))

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
                trained_obs = obs[0]
                training_obs = obs[1]
                current_trained_obs = trained_obs
                current_training_obs = training_obs

                trained_num_batches = len(current_trained_obs)//self.args.batch_size
                training_num_batches = len(current_training_obs)//self.args.batch_size

                if len(current_trained_obs) > self.args.batch_size*trained_num_batches:
                    trained_num_batches += 1
                if len(current_training_obs) > self.args.batch_size*training_num_batches:
                    training_num_batches += 1

                next_hidden_states_list = None
                next_cell_states_list = None
                hidden_states_list = None
                cell_states_list = None
                loss_batch = 0
                for j in range(self.args.time_step):
                    trained_actions = []
                    trained_ids = []
                    trained_action_batches = []
                    trained_view_batches = []
                    trained_view_ids = []
                    trained_view_values_list = []
                    trained_view_agent_embeddings_list = []
                    trained_hidden_states = []
                    trained_cell_states = []
                    for k in range(trained_num_batches):
                        view = current_trained_obs[k*self.args.batch_size:(k+1)*self.args.batch_size]

                        batch_id, batch_view, batch_agent_embeddings = self.process_view_with_emb_batch(view)
                        trained_view_agent_embeddings_list.append(batch_agent_embeddings)

                        ## Initial State: Zeros
                        if j == 0:
                            init_hidden_state, init_cell_state = self.q_net.init_hidden_states(len(view))
                            out, hidden_state , cell_state = self.trained_q_net(batch_view, batch_agent_embeddings,
                                                                        Variable(torch.from_numpy(init_hidden_state)).type(self.dtype),
                                                                        Variable(torch.from_numpy(init_cell_state)).type(self.dtype))
                        else:
                            out, hidden_state , cell_state = self.q_net(batch_view, batch_agent_embeddings, trained_hidden_states_list[k], trained_cell_states_list[k])

                        if np.random.rand() < eps_greedy:
                            action = out.max(1)[1].cpu().numpy()
                        else:
                            action = np.random.randint(self.num_actions, size=len(batch_view))

                        trained_ids.extend(batch_id)
                        trained_actions.extend(action)
                        trained_action_batches.append(action)
                        trained_view_batches.append(view)
                        trained_view_ids.append(batch_id)
                        trained_view_values_list.append(batch_view)
                        trained_hidden_states.append(hidden_state)
                        trained_cell_states.append(cell_state)
                    trained_hidden_states_list = trained_hidden_states
                    trained_cell_states_list = trained_cell_states

                    training_actions = []
                    training_ids = []
                    training_action_batches = []
                    training_view_batches = []
                    training_view_ids = []
                    training_view_values_list = []
                    training_view_agent_embeddings_list = []
                    training_hidden_states = []
                    training_cell_states = []
                    training_outs = []

                    for k in range(training_num_batches):
                        view = current_training_obs[k*self.args.batch_size:(k+1)*self.args.batch_size]

                        batch_id, batch_view, batch_agent_embeddings = self.process_view_with_emb_batch(view)
                        training_view_agent_embeddings_list.append(batch_agent_embeddings)

                        ## Initial State: Zeros
                        if j == 0:
                            init_hidden_state, init_cell_state = self.q_net.init_hidden_states(len(view))
                            out, hidden_state , cell_state = self.q_net(batch_view, batch_agent_embeddings,
                                                                        Variable(torch.from_numpy(init_hidden_state)).type(self.dtype),
                                                                        Variable(torch.from_numpy(init_cell_state)).type(self.dtype))
                        else:
                            out, hidden_state , cell_state = self.q_net(batch_view, batch_agent_embeddings, training_hidden_states_list[k], training_cell_states_list[k])

                        if np.random.rand() < eps_greedy:
                            action = out.max(1)[1].cpu().numpy()
                        else:
                            action = np.random.randint(self.num_actions, size=len(batch_view))

                        training_ids.extend(batch_id)
                        training_actions.extend(action)
                        training_action_batches.append(action)
                        training_view_batches.append(view)
                        training_view_ids.append(batch_id)
                        training_view_values_list.append(batch_view)
                        training_hidden_states.append(hidden_state)
                        training_cell_states.append(cell_state)
                        training_outs.append(out)

                    training_hidden_states_list = training_hidden_states
                    training_cell_states_list = training_cell_states

                    random_ids = list(self.env.random_agents.keys())
                    random_actions = np.random.randint(0, self.num_actions, len(random_ids))
                    actions = {**dict(zip(trained_ids, trained_actions)), **dict(zip(training_ids, training_actions)), **dict(zip(random_ids, random_actions))}

                    trained_env.take_actions(actions)

                    if self.args.time_step == j+1:
                        next_obs, rewards, killed = get_obs(trained_env)
                    else:
                        next_obs = get_obs(trained_env, only_view=True)
                    next_trained_obs, next_training_obs = next_obs


                    next_hidden_states = []
                    next_cell_states = []
                    for k in range(training_num_batches):
                        view_id = training_view_ids[k]
                        view_values = training_view_values_list[k]
                        next_view = next_training_obs[k*self.args.batch_size:(k+1)*self.args.batch_size]
                        next_view_id, next_view_values, next_agent_embeddings = self.process_view_with_emb_batch(next_view)
                        z, hidden_state, cell_state = self.q_net(view_values, training_view_agent_embeddings_list[k], training_hidden_states[k], training_cell_states[k])
                        hidden_states = training_hidden_states[k]
                        cell_states = training_cell_states[k]

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
                            q_value = z.gather(1, Variable(torch.Tensor(training_action_batches[k])).view(len(view_values), 1).type(self.dlongtype))
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

                    current_training_obs = next_training_obs
                    current_trained_obs = next_trained_obs
                    next_hidden_states_list = next_hidden_states
                    next_cell_states_list = next_cell_states

                trained_actions = []
                trained_action_batches = []
                trained_ids = []
                #obs = get_obs(self.env, only_view=True)
                for k in range(trained_num_batches):
                    view = trained_obs[k*self.args.batch_size:(k+1)*self.args.batch_size]

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

                    trained_actions.extend(action)
                    trained_action_batches.append(action)
                    trained_ids.extend(batch_id)

                training_actions = []
                training_action_batches = []
                training_ids = []
                #obs = get_obs(self.env, only_view=True)
                for k in range(training_num_batches):
                    view = training_obs[k*self.args.batch_size:(k+1)*self.args.batch_size]

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

                    training_actions.extend(action)
                    training_action_batches.append(action)
                    training_ids.extend(batch_id)

                random_ids = list(self.env.random_agents.keys())
                random_actions = np.random.randint(0, self.num_actions, len(random_ids))
                actions = {**dict(zip(trained_ids, trained_actions)), **dict(zip(training_ids, training_actions)), **dict(zip(random_ids, random_actions))}
                self.env.take_actions(actions)

                _, rewards, killed = get_obs(self.env)
                episode_reward = np.sum(list(rewards.values()))
                total_reward += episode_reward
                self.env.killed = killed

                predator_rewards, prey_rewards = self.env.return_total_reward(rewards)
                #writer.add_scalar('predator_reward', predator_rewards, timesteps)
                #writer.add_scalar('prey_reward', prey_rewards, timesteps)

                increase_predators = self.env.increase_predators
                increase_preys = self.env.increase_preys
                killed = self.env.remove_dead_agents()
                self.remove_dead_agent_emb(killed)


                loss += (loss_batch/(j+1))
                msg = "episode {:03d} episode step {:03d} loss:{:5.4f} reward:{:5.3f} eps_greedy {:5.3f}".format(episode, i, loss_batch/(j+1), episode_reward/len(obs), eps_greedy)
                bar.set_description(msg)
                bar.update(1)


                if self.args.env_type == 'simple_population_dynamics':
                    info = "Step\t{:03d}\tReward\t{:5.3f}\tnum_agents\t{:d}\tnum_random_preys\t{:d}\tnum_trained_preys\t{:d}\tnum_training_preys\t{:d}\tnum_ramdom_predators\t{:d}\tnum_trained_predators\t{:d}\tnum_training_predators\t{:d}\tincrease_predators\t{:d}\tincrease_preys\t{:d}\tkilled_agents\t{:d}".format(i, episode_reward/len(obs), len(self.env.agents), len(self.env.random_preys), len(self.env.trained_preys), len(self.env.training_preys), len(self.env.random_predators), len(self.env.trained_predators), len(self.env.training_predators),increase_predators, increase_preys, len(killed))
                    log.write(info+'\n')
                    log.flush()
                    timesteps += 1
                    if i % self.args.increase_every == 0:
                        self.env.increase_prey(self.args.prey_increase_prob)
                        self.env.increase_predator(self.args.predator_increase_prob)
                elif self.args.env_type == 'complex_population_dynamics':
                    info = "Step\t{:03d}\tReward\t{:5.3f}\tnum_agents\t{:d}\tnum_preys\t{:d}\tnum_predators\t{:d}\tincrease_predators\t{:d}\tincrease_preys\t{:d}\tkilled_agents\t{:d}\taverage_predator_health\t{:5.3f}\taverage_prey_health\t{:5.3f}\taverage_predator_attack\t{:5.3f}\taverage_prey_attack\t{:5.3f}\taverage_predator_resilience\t{:5.3f}\taverage_prey_resilience\t{:5.3f}\taverage_predator_speed\t{:5.3f}\taverage_prey_speed\t{:5.3f}".format(i, episode_reward/len(obs), len(self.env.agents), len(self.env.preys), len(self.env.predators),increase_predators, increase_preys, len(killed), np.mean(self.env.predator_health), np.mean(self.env.prey_health), np.mean(self.env.predator_attack), np.mean(self.env.prey_attack), np.mean(self.env.predator_resilience), np.mean(self.env.prey_resilience), np.mean(self.env.predator_speed), np.mean(self.env.prey_speed))
                    log.write(info+'\n')
                    log.flush()
                    timesteps += 1
                    self.env.crossover_prey(self.args.crossover_scope, crossover_rate=self.args.prey_increase_prob, mutation_prob=self.args.mutation_prob)
                    self.env.crossover_predator(self.args.crossover_scope, crossover_rate=self.args.predator_increase_prob, mutation_prob=self.args.mutation_prob)
                    if i % self.args.increase_every == 0:
                        self.env.add_preys(1)
                        self.env.add_predators(1)
                elif self.args.env_type != 'simple_population_dynamics_ga_action':
                    info = "Step\t{:03d}\tReward\t{:5.3f}\tnum_agents\t{:d}\tnum_random_preys\t{:d}\tnum_trained_preys\t{:d}\tnum_training_preys\t{:d}\tnum_ramdom_predators\t{:d}\tnum_trained_predators\t{:d}\tnum_training_predators\t{:d}\tincrease_predators\t{:d}\tincrease_preys\t{:d}\tkilled_agents\t{:d}".format(i, episode_reward/len(obs), len(self.env.agents), len(self.env.random_preys), len(self.env.trained_preys), len(self.env.training_preys), len(self.env.random_predators), len(self.env.trained_predators), len(self.env.training_predators),increase_predators, increase_preys, len(killed))
                    log.write(info+'\n')
                    log.flush()
                    timesteps += 1
                    self.env.crossover_prey(self.args.crossover_scope, crossover_rate=self.args.prey_increase_prob)
                    self.env.crossover_predator(self.args.crossover_scope, crossover_rate=self.args.predator_increase_prob)
                    self.env.add_preys(1)
                    self.env.add_predators(1)
                if len(self.env.predator_agents) < 2 or len(self.env.prey_agents) < 2 or len(self.env.prey_agents) > self.args.prey_capacity or len(self.env.predator_agents) > self.args.predator_capacity:
                    log.close()
                    break
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

    def create_dir(self, rounds):
        if self.experiment_type == 'variation':
            img_dir = os.path.join('results', self.args.env_type, 'exp_{:d}'.format(self.args.experiment_id), 'variation_images', str(self.args.variation_id), str(rounds))
            log_dir = os.path.join('results', self.args.env_type, 'exp_{:d}'.format(self.args.experiment_id), 'variation_logs', str(self.args.variation_id), str(rounds))
        else:
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
        return img_dir, log_dir

    def create_model_dir(self, model_dir):
        model_dir = os.path.join('results', self.args.env_type, 'exp_{:d}'.format(self.args.experiment_id), 'models')
        #writer i SummaryWriter(log_dir=os.path.join('results', self.args.env_type, 'exp_{:d}'.format(self.args.experiment_id)))
        try:
            os.makedirs(model_dir)
        except:
            shutil.rmtree(model_dir)
            os.makedirs(model_dir)
        return model_dir


    def take_random_action(self, agents):
        actions = np.random.randint(self.num_actions, len(agents))



    def count_agents_in_local_region(self, region_size=50):
        n = self.env.h // region_size
        for i in range(n):
            local = self.env.map[i*region_size:(i+1)*region_size, i*region_size:(i+1)*region_size]
            agent_indices = np.where(local > 0)




