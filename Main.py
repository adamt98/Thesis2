from Environments import DiscreteEnv
from Generators import GBM_Generator
import Models
import Utils
import numpy as np
import matplotlib.pyplot as plt

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
        figure = plt.figure()

        obs = self.test_env.reset()
        done = False
        while not done:
            action, _states = self.model.predict(obs, deterministic=False)
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

##### Environment config ###############

sigma = 0.01*np.sqrt(250) # 1% vol per day, annualized
r = 0.0 # Annualized
S0 = 100
freq = 0.2 #0.2 corresponds to trading freq of 5x per day
ttm = 50 # 50 & freq=0.2 => 10 days expiry
kappa = 0.8
cost_multiplier = 0.0
discount = 0.88

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
n_sim = 100
observe_dim = 3
action_num = 101

max_episodes = 6000

epoch = 3000 # roll out 3000 episodes, then train
n_epochs = 15 # 5 <=> pass over the rollout 5 times
batch_size = 30

policy_kwargs = dict(#activation_fn=torch.nn.ReLU,
                     net_arch=[32,32, dict(pi=[32,32], vf=[32])]) # dict(pi=[10], vf=[10])

gradient_max = 1.0
gae_lambda = 0.96
value_weight = 1.0
entropy_weight = 0.05

def lr(x : float): 
    return 1e-5 + (5e-4-1e-5)*x
#lr=3e-5
surrogate_loss_clip = 0.1 # min and max acceptable KL divergence

def simulate(env, obs):
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=False)
        obs, reward, done, info = env.step(action)
    
    return info['output']

if __name__ == "__main__":

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
    model.save('./weights_PPO/')

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
    Utils.plot_pnl(delta, df)

    pnl_paths_dict, pnl_dict, tcosts_dict, ntrades_dict = Utils.simulate_pnl(delta_agent, n_sim, test_env, simulate)
    Utils.plot_pnl_hist(pnl_paths_dict, pnl_dict, tcosts_dict, ntrades_dict)
