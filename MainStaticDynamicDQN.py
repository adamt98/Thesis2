# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.0
# ---

from cProfile import label
from math import gamma
from re import T
from typing import Callable

import gym
from Environments2 import BarrierEnv, BarrierEnv2, BarrierEnv3
from Generators import GBM_Generator
import Models
import Utils
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import torch
from torch.utils.tensorboard import SummaryWriter
#writer = SummaryWriter()
from Models import DeltaHedge

from stable_baselines3 import PPO, DQN
import Utils

import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Figure
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.vec_env import VecMonitor

class FigureRecorderCallback(BaseCallback):
    def __init__(self, test_env, verbose=0):
        super(FigureRecorderCallback, self).__init__(verbose)
        self.test_env = test_env

    def _on_step(self): return True

    def _on_rollout_end(self):
        figure = plt.figure()

        obs = self.test_env.reset()
        done = False
        while not done:
            action, _states = self.model.predict(obs, deterministic=True)
            obs, reward, done, info = self.test_env.step(action)
        
    
        model_actions = info['output'].actions.values

        # delta hedge benchmark
        obs = self.test_env.reset()
        delta_agent = DeltaHedge(self.test_env.generator.initial, n_puts_sold=n_puts_sold, min_action=min_action)
        delta_actions = delta_agent.test(self.test_env, obs).actions.values

        figure.add_subplot().plot(delta_actions, 'b-', model_actions, 'g-')
        
        # Close the figure after logging it
        self.logger.record("trajectory/figure", Figure(figure, close=True), exclude=("stdout", "log", "json", "csv"))
        plt.close()
        return True

def make_env(env_args, rank: int, seed: int = 0) -> Callable:
    """
    Utility function for multiprocessed env.
    
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    :return: (Callable)
    """
    def _init() -> gym.Env:
        env = BarrierEnv3(**env_args)
        env.seed(seed + rank)
        return env
        
    return _init

# #### Environment config ###############

sigma = 0.01*np.sqrt(250) # 1% vol per day, annualized
r = 0.0 # Annualized
S0 = 100
freq = 0.2 #0.2 corresponds to trading freq of 5x per day
ttm = 50 # 50 & freq=0.2 => 10 days expiry
kappa = 1.0
cost_multiplier = 0.0
discount = 0.85

barrier = 97
n_puts_sold = 0
min_action = 0
max_action = 400
action_num = max_action - min_action
max_sellable_puts = 5

generator = GBM_Generator(S0, r, sigma, freq, barrier=barrier)
env_args = {
    "generator" : generator,
    "ttm" : ttm,
    "kappa" : kappa,
    "cost_multiplier" : cost_multiplier,
    "reward_type" : "static",
    "testing" : False,
    #"n_puts_sold" : n_puts_sold,
    "min_action" : min_action,
    "max_action" : max_action,
    "max_sellable_puts" : max_sellable_puts
}


num_cpu = 7

##########################################
##### PPO Training hyperparameter setup ######
n_sim = 100
observe_dim = 4 #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


max_episodes = 50*40

#epoch = int(50*240/7) # roll out 3000 episodes, then train
#n_epochs = 5 # 5 <=> pass over the rollout 5 times
batch_size = 30

policy_kwargs = dict(activation_fn=torch.nn.ReLU, # Du uses ReLU
                     net_arch=[50,50,50,50])

gradient_max = 1.0

buffer_size = 50*15000
lstart = 50*1000 # after 1000 episodes
train_freq = 50*30 # update policy net every 50 timesteps
grad_steps = 50*30 # default 1
target_update_interval = 50*5000 # default 10000 timesteps

def lr(x : float): 
    return 1e-4 #1e-5 + (1e-4-1e-5)*x


def simulate(env, obs):
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
    
    return info['output']

if __name__ == "__main__":
    env = VecMonitor(SubprocVecEnv([make_env(env_args, i) for i in range(num_cpu)]))

    model = DQN(policy="MlpPolicy",
                policy_kwargs=policy_kwargs,
                env=env,
                learning_rate=lr,
                buffer_size=buffer_size,
                learning_starts=lstart,
                batch_size=batch_size,
                gamma = discount,
                train_freq=train_freq,
                gradient_steps=grad_steps,
                target_update_interval=target_update_interval,
                exploration_fraction=0.9,
                exploration_initial_eps=1.0,
                exploration_final_eps=0.05,
                max_grad_norm=gradient_max,
                tensorboard_log='./runsDQN/',
                verbose=1
                )

    generator = GBM_Generator(S0, r, sigma, freq, seed=123, barrier=barrier)
    test_env_args = {
        "generator" : generator,
        "ttm" : ttm,
        "kappa" : kappa,
        "cost_multiplier" : cost_multiplier,
        "reward_type" : "static",
        "testing" : True,
        #"n_puts_sold" : n_puts_sold,
        "min_action" : min_action,
        "max_action" : max_action,
        "max_sellable_puts" : max_sellable_puts
    }

    test_env = BarrierEnv3(**test_env_args)

    model.learn(total_timesteps=max_episodes , callback=FigureRecorderCallback(test_env))
    model.save('./weights_DQN/')

    #####################################
    ####### TESTING PHASE ###############
    #####################################
    
    obs = test_env.reset()
    df = simulate(test_env, obs)
    # delta hedge benchmark
    delta_agent = DeltaHedge(generator.initial, call = False, n_puts_sold=n_puts_sold, min_action=min_action)
    obs = test_env.reset()
    delta = delta_agent.test(test_env, obs)

    Utils.plot_decisions(delta, df)

    outPPO = Utils.simulate_pnl("DQN", n_sim, test_env, simulate)
    outDelta = Utils.simulate_pnl("Delta", n_sim, test_env, delta_agent.test)
    out = pd.concat([outPPO, outDelta], ignore_index=True)
    Utils.plot_pnl_hist(out)
    Utils.perf_measures(out)
    js = Utils.getJSDivergence(outPPO, outDelta)
    print("JS divergence table:")
    print(js.head())