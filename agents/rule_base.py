import os, sys

import numpy as np
# from garl_gym.scenarios.simple_population_dynamics_rule_base import *
from tqdm import tqdm


def run_rulebase(env, args):
    model_dir = os.path.join('results', 'exp_{:d}'.format(args.experiment_id), 'models')
    bar = tqdm(range(args.test_step))
    try:
        os.makedirs(model_dir)
    except:
        shutil.rmtree(model_dir)
        os.makedirs(model_dir)
    rounds = 0

    img_dir = os.path.join('results', 'exp_{:d}'.format(args.experiment_id), 'images',str(rounds))
    log_dir = os.path.join('results', 'exp_{:d}'.format(args.experiment_id), 'logs', str(rounds))
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


    for i in range(args.test_step):
        obs = get_obs(env, only_view=True)
        if args.video_flag:
            env.dump_image(os.path.join(img_dir, '{:d}.png'.format(i)))
        env.take_actions()

        next_view_batches, rewards, killed = get_obs(env)
        env.killed = killed
        episode_reward = np.sum(list(rewards.values()))


        killed = env.remove_dead_agents()

        view_batches = next_view_batches
        msg = "episode {:03d} episode step {:03d} reward:{:5.3f}".format(0, i, episode_reward/len(obs))
        bar.set_description(msg)
        bar.update(1)

        info = "Episode\t{:03d}\tStep\t{:03d}\tReward\t{:5.3f}\tnum_agents\t{:d}\tnum_preys\t{:d}\tnum_predators\t{:d}".format(0, i, episode_reward/len(obs), len(env.agents), len(self.env.preys), len(self.env.predators))
        log.write(info+'\n')
        log.flush()

        env.crossover_prey(self.args.crossover_scope, crossover_rate=self.args.prey_increase_prob)
        #env.increase_prey(self.args.prey_increase_prob)
        env.crossover_predator(self.args.crossover_scope, crossover_rate=self.args.predator_increase_prob)

    log.close()
