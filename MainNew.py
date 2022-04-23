from discrete_environments import DiscreteEnv, DiscreteEnv2
from data_generators import GBM_Generator, HestonGenerator
import utils
import numpy as np
import matplotlib.pyplot as plt

from machin.frame.algorithms import DQN
from machin.frame.algorithms.dqn import DQN
from machin.utils.logging import default_logger as logger
import torch as t
import torch.nn as nn
import gym
import tqdm

observe_dim = 3
action_num = 101
max_episodes = 30
solved_reward = -40
solved_repeat = 7


S0 = 100

# Annualized
sigma = 0.01*np.sqrt(250) # 1% vol per day
r = 0.0

freq = 0.2 # corresponds to trading freq of 5x per day
ttm = 50
kappa = 0.1
cost_multiplier = 0.0
gamma = 0.85

generator = GBM_Generator(S0, r, sigma, freq)

env_args = {
    "generator" : generator,
    "ttm" : ttm,
    "kappa" : kappa,
    "cost_multiplier" : cost_multiplier,
    "testing" : False
}

env = DiscreteEnv2(**env_args)
#drl_env, _ = env.get_sb_env()

# 1 epoch = 3000 episodes = 150k time-steps
epoch = 150000
batch_size = 32
n_epochs_per_update = 5

#n_updates = int(epoch * n_epochs_per_update / batch_size)
final_eps = 0.05
eps_decay = np.exp(np.log(final_eps)/(max_episodes*50))

import models

layers = [8, 16, 32, 64]

## test barrier opts
# spot = []
# barr = []
# for i in range(100):
#     gen = GBM_Generator(50+i, 0.0, 0.2, 1, barrier=130)
#     is_knocked = 130 > (50 + i)
#     tmp = gen.get_barrier_value(120, 250, up=False, out=True, is_knocked=is_knocked, call=True)
#     barr.append(tmp)
#     spot.append(50+i)

# plt.plot(spot, barr)
# plt.show()

if __name__ == "__main__":
    # dqn = models.DQN_Model(observe_dim, action_num, layers, eps_decay, learning_rate=0.00001, batch_size=batch_size, discount=0.9)
    # dqn.train(max_episodes=max_episodes, env=env, solved_reward=solved_reward, solved_repeat=solved_repeat)

    # generator = GBM_Generator(S0, r, sigma, freq)
    # env_args = {
    #     "generator" : generator,
    #     "ttm" : ttm,
    #     "kappa" : kappa,
    #     "cost_multiplier" : cost_multiplier,
    #     "testing" : True
    # }
    # dqn.test(generator=generator, env_args=env_args, n_sim=300)

    ## TESTING PPO
    env_args = {
        "generator" : generator,
        "ttm" : ttm,
        "kappa" : kappa,
        "cost_multiplier" : cost_multiplier,
        "testing" : False
    }

    env = DiscreteEnv2(**env_args)

    ppo = models.PPO_Model(observe_dim, action_num, layers, learning_rate=0.00001, batch_size=batch_size, discount=0.9)
    ppo.train(max_episodes=max_episodes, env=env, solved_reward=solved_reward, solved_repeat=solved_repeat)

    generator = GBM_Generator(S0, r, sigma, freq)
    env_args = {
        "generator" : generator,
        "ttm" : ttm,
        "kappa" : kappa,
        "cost_multiplier" : cost_multiplier,
        "testing" : True
    }

    ppo.test(generator=generator, env_args=env_args, n_sim=30)



    ## TESTING MODEL AVERAGER
    # n_steps = 500
    # n_batches = 3

    # eps_func = utils.EpsFunction(n_steps).get_func()

    # env = DiscreteEnv(**env_args)
    # drl_env, _ = env.get_sb_env()
    # agent = models.ModelAverager(env, gamma)


    # agent.train(n_steps, n_batches, eps_func)

    # generator = GBM_Generator(S0, r, sigma, freq)
    # env_args = {
    #     "generator" : generator,
    #     "ttm" : ttm,
    #     "kappa" : kappa,
    #     "cost_multiplier" : cost_multiplier,
    #     "testing" : True
    # }
    # agent.test(generator, env_args, n_sim=50)