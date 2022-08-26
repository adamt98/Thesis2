from Environments import DiscreteEnv
from Generators import GBM_Generator
from Models import DeltaHedge
import Utils

from typing import Callable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gym
import torch

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Figure
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

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
        delta_agent = DeltaHedge(self.test_env.generator.initial)
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
        env = DiscreteEnv(**env_args)
        env.seed(seed + rank)
        return env
        
    return _init
    
##### Environment config ###############

sigma = 0.01*np.sqrt(250) # 1% vol per day, annualized
r = 0.0 # Annualized
S0 = 100
freq = 0.2 #0.2 corresponds to trading freq of 5x per day
ttm = 50 # 50 & freq=0.2 => 10 days expiry
kappa = 0.1 # .3
cost_multiplier = 0.5
discount = 0.85

generator = GBM_Generator(S0, r, sigma, freq)
env_args = {
    "generator" : generator,
    "ttm" : ttm,
    "kappa" : kappa,
    "cost_multiplier" : cost_multiplier,
    "reward_type" : "basic",
    "testing" : False
}

env = DiscreteEnv(**env_args)

##########################################
##### PPO Training hyperparameter setup ######
# n_sim = 100
# observe_dim = 3
# action_num = 101

# max_episodes = 50*12000

# epoch = 3000 # roll out 3000 episodes, then train
# n_epochs = 5 # 5 <=> pass over the rollout 5 times
# batch_size = 30

# policy_kwargs = dict(activation_fn=torch.nn.Tanh,
#                      net_arch=[dict(pi=[20,20,20], vf=[40,40])]) 

# gradient_max = 1.0
# gae_lambda = 0.9
# value_weight = 0.8
# entropy_weight = 0.1
# surrogate_loss_clip = 0.25 # min and max acceptable KL divergence

##########################################
##### DQN Training hyperparameter setup ######
num_cpu=7
n_sim = 100
observe_dim = 3
action_num = 101

max_episodes = 50*38001

#epoch = int(50*3000) # roll out 3000 episodes, then train
#n_epochs = 2 # 5 <=> pass over the rollout 5 times
batch_size = 30


policy_kwargs = dict(activation_fn=torch.nn.ReLU, # Du uses ReLU
                     net_arch=[30,30,30,30])

gradient_max = 1.0

buffer_size = 50*15000
lstart = 50*800 # after 1000 episodes
train_freq = 50 # update policy net every 100 episodes
grad_steps = 50 # default 1
target_update_interval = 50*1000 # default 3000 episodes

def lr(x : float): 
    return 1e-5 + (7e-5 - 1e-5)*x


def simulate(env, obs):
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
    
    return info['output']

if __name__ == "__main__":

    # model = PPO(policy="MlpPolicy", 
    #             policy_kwargs=policy_kwargs,
    #             env=env,
    #             learning_rate=lr, 
    #             n_steps = epoch,
    #             batch_size=batch_size,
    #             n_epochs=n_epochs,
    #             gamma = discount,
    #             gae_lambda=gae_lambda,
    #             clip_range=surrogate_loss_clip,
    #             normalize_advantage=True,
    #             ent_coef=entropy_weight,
    #             vf_coef=value_weight,
    #             max_grad_norm=gradient_max,
    #             tensorboard_log='./runs/',
    #             verbose=1)
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

    generator = GBM_Generator(S0, r, sigma, freq, seed=123)
    test_env_args = {
        "generator" : generator,
        "ttm" : ttm,
        "kappa" : kappa,
        "cost_multiplier" : cost_multiplier,
        "reward_type" : "basic",
        "testing" : True
    }

    test_env = DiscreteEnv(**test_env_args)

    model.learn(total_timesteps=max_episodes, callback=FigureRecorderCallback(test_env))
    model.save('./weights_DQN/')

    #####################################
    ####### TESTING PHASE ###############
    #####################################
    
    obs = test_env.reset()
    df = simulate(test_env, obs)
    # delta hedge benchmark
    delta_agent = DeltaHedge(generator.initial)
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
