from cProfile import label
from re import T
from typing import Callable

import gym
from Environments2 import BarrierEnv, BarrierEnv2, BarrierEnv4, BarrierEnv3
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

from stable_baselines3 import PPO
import Utils

import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Figure

class FigureRecorderCallback(BaseCallback):
    def __init__(self, test_env, verbose=0):
        super(FigureRecorderCallback, self).__init__(verbose)
        self.test_env = test_env

    def _on_step(self): return True

    def _on_rollout_end(self):
        
        obs = self.test_env.reset()
        done = False
        while not done:
            action, _states = self.model.predict(obs, deterministic=True)
            obs, reward, done, info = self.test_env.step(action)
        
    
        model_actions = info['output'].actions.values
        #model_actions_opt = info['output'].actions_opt.values
        # delta hedge benchmark
        obs = self.test_env.reset()
        delta_agent = DeltaHedge(self.test_env.generator.initial, n_puts_sold=n_puts_sold, min_action=min_action, put_K = put_strike)
        delta_out = delta_agent.test(self.test_env, obs)
        delta_actions = delta_out.actions.values
        #delta_actions_opt = delta_out.actions_opt.values

        figure = plt.figure()
        figure.add_subplot().plot(delta_actions, 'b-', model_actions, 'g-')
        self.logger.record("trajectory/figure/und", Figure(figure, close=True), exclude=("stdout", "log", "json", "csv"))
        plt.close()

        # figure = plt.figure()
        # figure.add_subplot().plot(delta_actions_opt, 'b-', model_actions_opt, 'g-')
        # self.logger.record("trajectory/figure/opt", Figure(figure, close=True), exclude=("stdout", "log", "json", "csv"))
        # plt.close()

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
        env = BarrierEnv2(**env_args)
        env.seed(seed + rank)
        return env
        
    return _init

# #### Environment config ###############

sigma = 0.35 #0.01*np.sqrt(250) # 1% vol per day, annualized
r = 0.0 # Annualized
S0 = 100
freq = 0.2 #0.2 corresponds to trading freq of 5x per day
ttm = 50 # 50 & freq=0.2 => 10 days expiry
kappa = 0.1
cost_multiplier = 0.3
discount = 0.9

put_strike = 100

barrier = 97
n_puts_sold = 1 # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
min_action = -100
max_action = 300
action_num = max_action - min_action
#max_sellable_puts=1

generator = GBM_Generator(S0, r, sigma, freq, barrier=barrier)
env_args = {
    "generator" : generator,
    "ttm" : ttm,
    "kappa" : kappa,
    "cost_multiplier" : cost_multiplier,
    "reward_type" : "static",
    "testing" : False,
    "n_puts_sold" : n_puts_sold,
    "min_action" : min_action,
    "max_action" : max_action,
    #"max_sellable_puts" : max_sellable_puts,
    #"put_K" : put_strike
}

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.vec_env import VecMonitor
num_cpu = 7

##########################################
##### PPO Training hyperparameter setup ######
n_sim = 100
observe_dim = 4 # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


max_episodes = 50*25000

epoch = int(50*300/7) # roll out 240 episodes, then train
n_epochs = 5 # 5 <=> pass over the rollout 5 times
batch_size = 30

policy_kwargs = dict(activation_fn=torch.nn.Tanh,
                     net_arch=[dict(pi=[20,20,20,20], vf=[50,50])])

gradient_max = 1.0
gae_lambda = 0.9
value_weight = 0.6
entropy_weight = 0.13

def surrogate_loss_clip(x : float):
    return 0.25 #0.15 + (0.35 - 0.15)*x  

def lr(x : float): 
    return 1e-4 + (1e-3 - 1e-4)*x


def simulate(env, obs):
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
    
    return info['output']

from stable_baselines3.common.env_checker import check_env

if __name__ == "__main__":
    generator = GBM_Generator(100, 0, 0.16, freq, barrier=barrier)
    hits = 0
    total = 1000
    for i in range(total):
        generator.reset_with_seed(i)
        for j in range(150):
            if generator.get_next() <= 97:
                hits += 1
                break

    print(hits/total)
    print("done")
    # vals = []
    # vals2 = []
    # deltas = []
    # delta2 = []
    # spots = np.arange(80, 120, 0.05)
    # for spot in spots:
    #     ttm1 = 50
    #     barrier = 97
    #     K = 100
    #     K1 = 97.0
    #     generator = GBM_Generator(spot, r, sigma, freq, barrier=barrier)
    #     val = generator.get_barrier_value(K=K, ttm=ttm1, up=False, out=False, call=False)
    #     val2 = generator.get_option_value(K=K1, ttm=ttm1, call=False)
    #     delta_barr = generator.get_DIP_vega(spot, K=K, ttm=ttm1) 
        
    #     delta_static = generator.get_vega(spot, K1, ttm1)# - 1 # get_delta gives call delta, we're selling a put
    #     delta2.append(delta_static)
        
    #     vals.append(val)
    #     vals2.append(val2)
    #     deltas.append(delta_barr)

    # plt.plot(spots, deltas, label = "DIP")
    # plt.plot(spots, delta2, label = "Vanilla put")
    # plt.legend()
    # plt.title("Delta comparison")
    # #plt.savefig('deltas_comparison')
    # plt.show()

    # plt.plot(spots, vals, label = "DIP")
    # plt.plot(spots, vals2, label = "Vanilla put")
    # plt.legend()
    # plt.title("Value comparison")
    # #plt.savefig('vals_comparison')
    # plt.show()

    # diffs = np.array(deltas) - np.array(delta2)
    # plt.plot(spots, diffs, label = "DIP - Vanilla put")
    # plt.legend()
    # plt.title("Difference in deltas")
    # #plt.savefig('delta_diff')
    # plt.show()

    # vals3 = []
    # unds = []
    # generator = GBM_Generator(100, r, sigma, freq)
    # for i in range(100):
    #     unds.append(generator.get_next())
    #     vals3.append(generator.get_option_value(100,100-i,False))

    # plt.figure(1)
    # plt.subplot(121)
    # plt.plot(vals3)
    # plt.subplot(122)
    # plt.plot(unds)
    # plt.show()

    env = VecMonitor(SubprocVecEnv([make_env(env_args, i) for i in range(num_cpu)]))
    #env = BarrierEnv3(**env_args)
    model = PPO(policy="MlpPolicy", 
                policy_kwargs=policy_kwargs,
                env=env,
                learning_rate=lr, 
                n_steps = epoch,
                batch_size=batch_size,
                n_epochs=n_epochs,
                gamma = discount,
                gae_lambda=gae_lambda,
                clip_range=surrogate_loss_clip,
                normalize_advantage=True,
                ent_coef=entropy_weight,
                vf_coef=value_weight,
                max_grad_norm=gradient_max,
                tensorboard_log='./runs/',
                verbose=1)

    generator = GBM_Generator(S0, r, sigma, freq, seed=123, barrier=barrier)
    test_env_args = {
        "generator" : generator,
        "ttm" : ttm,
        "kappa" : kappa,
        "cost_multiplier" : cost_multiplier,
        "reward_type" : "static",
        "testing" : True,
        "n_puts_sold" : n_puts_sold,
        "min_action" : min_action,
        "max_action" : max_action,
        #"max_sellable_puts" : max_sellable_puts,
        #"put_K" : put_strike
    }

    test_env = BarrierEnv2(**test_env_args)

    model.learn(total_timesteps=max_episodes, callback=FigureRecorderCallback(test_env)) # 
    model.save('./weights_PPO/')

    #####################################
    ####### TESTING PHASE ###############
    #####################################
    
    obs = test_env.reset()
    df = simulate(test_env, obs)
    # delta hedge benchmark
    delta_agent = DeltaHedge(generator.initial, call = False, n_puts_sold=n_puts_sold, min_action=min_action, put_K = put_strike)
    obs = test_env.reset()
    delta = delta_agent.test(test_env, obs)

    Utils.plot_decisions(delta, df)
    #Utils.plot_decisions_extra(delta, df)

    outPPO = Utils.simulate_pnl("PPO", n_sim, test_env, simulate)
    outDelta = Utils.simulate_pnl("Delta", n_sim, test_env, delta_agent.test)
    out = pd.concat([outPPO, outDelta], ignore_index=True)
    Utils.plot_pnl_hist(out)
    Utils.perf_measures(out)
    js = Utils.getJSDivergence(outPPO, outDelta)
    print("JS divergence table:")
    print(js.head())