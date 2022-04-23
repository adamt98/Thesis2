from agents import ModelAverager, DeltaHedge, DRLAgent
from discrete_environments import DiscreteEnv, DiscreteEnv2
from data_generators import GBM_Generator, HestonGenerator
import utils
import numpy as np
import matplotlib.pyplot as plt

from machin.frame.algorithms import PPO
from machin.frame.algorithms.ppo import PPO
from machin.utils.logging import default_logger as logger
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import gym

observe_dim = 3
action_num = 101
max_episodes = 15000
solved_reward = 0
solved_repeat = 10


# model definition
class Actor(nn.Module):
    def __init__(self, state_dim, action_num):
        super().__init__()

        self.fc1 = nn.Linear(state_dim, 16)
        self.bn1 = nn.LayerNorm(16, elementwise_affine=False)
        self.fc2 = nn.Linear(16, 32)
        self.bn2 = nn.LayerNorm(32, elementwise_affine=False)
        self.fc3 = nn.Linear(32, 64)
        self.bn3 = nn.LayerNorm(64, elementwise_affine=False)
        self.fc4 = nn.Linear(64, action_num)

    def forward(self, state, action=None):
        a = self.fc1(state)
        a = t.relu(self.bn1(a))
        a = self.fc2(a)
        a = t.relu(self.bn2(a))
        a = self.fc3(a)
        a = t.relu(self.bn3(a))
        a = self.fc4(a)

        probs = t.softmax(a, dim=1)
        dist = Categorical(probs=probs)
        act = action if action is not None else dist.sample()
        act_entropy = dist.entropy()
        act_log_prob = dist.log_prob(act.flatten())
        return act, act_log_prob, act_entropy

class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()

        self.fc1 = nn.Linear(state_dim, 16)
        self.bn1 = nn.LayerNorm(16, elementwise_affine=False)
        self.fc2 = nn.Linear(16, 16)
        self.bn2 = nn.LayerNorm(16, elementwise_affine=False)
        self.fc3 = nn.Linear(16, 16)
        self.bn3 = nn.LayerNorm(16, elementwise_affine=False)
        self.fc4 = nn.Linear(16, 1)

    def forward(self, state):
        a = self.fc1(state)
        a = t.relu(self.bn1(a))
        a = self.fc2(a)
        a = t.relu(self.bn2(a))
        a = self.fc3(a)
        a = t.relu(self.bn3(a))
        a = self.fc4(a)
        return a

S0 = 100

# Annualized
sigma = 0.01*np.sqrt(250) # 1% vol per day
r = 0.0

freq = 0.2 # corresponds to trading freq of 5x per day
ttm = 50
kappa = 0.1
cost_multiplier = 0.0
gamma = 0.85

batch_size = 32

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

actor = Actor(observe_dim, action_num)
critic = Critic(observe_dim)

model = PPO(actor,critic,
                t.optim.Adam,
                nn.MSELoss(reduction='sum'),
                visualize=False,
                learning_rate=1e-5,
                batch_size=batch_size,
                discount=gamma,
                gradient_max=1.0)

episode, step, reward_fulfilled = 0, 0, 0
smoothed_total_reward = 0
terminal = False


if __name__ == "__main__":
    while episode < max_episodes:
        episode += 1
        total_reward = 0
        terminal = False
        step = 0
        state = t.tensor(env.reset(), dtype=t.float32).view(1, observe_dim)
        ep_list = []
        while not terminal:
            step += 1
            with t.no_grad():
                old_state = state
                # agent model inference
                action = model.act(
                    {"state": old_state}
                )[0] # THIS IS ADDED!!!

                state, reward, terminal, _ = env.step(action.item())
                state = t.tensor(state, dtype=t.float32).view(1, observe_dim)
                total_reward += reward
                
                ep_list.append({
                    "state": {"state": old_state},
                    "action": {"action": action},
                    "next_state": {"state": state},
                    "reward": reward,
                    "terminal": terminal
                })

        model.store_episode(ep_list)
        model.update()

        # show reward
        smoothed_total_reward = (smoothed_total_reward * 0.9 +
                                    total_reward * 0.1)
        logger.info("Episode {} total reward={:.2f}"
                    .format(episode, smoothed_total_reward))

        if smoothed_total_reward > solved_reward:
            reward_fulfilled += 1
            if reward_fulfilled >= solved_repeat:
                logger.info("Environment solved!")
                break
        else:
            reward_fulfilled = 0

    # Testing
    generator = GBM_Generator(S0, r, sigma, freq, seed = 1234)
    env_args = {
    "generator" : generator,
    "ttm" : ttm,
    "kappa" : kappa,
    "cost_multiplier" : cost_multiplier,
    "testing" : True
    }

    env = DiscreteEnv2(**env_args)
    state = t.tensor(env.reset(), dtype=t.float32).view(1, observe_dim)
    terminal = False
    while not terminal:
        with t.no_grad():
            old_state = state
            # agent model inference
            action = model.act(
                {"state": old_state}
            )[0]
            state, reward, terminal, info = env.step(action.item())
            state = t.tensor(state, dtype=t.float32).view(1, observe_dim)
            
    df = info['output']

    # delta hedge benchmark
    test_env_delta = DiscreteEnv(**env_args)
    delta_agent = DeltaHedge(r, sigma, S0)
    delta = delta_agent.test(test_env_delta)

    utils.plot_decisions(delta, df)
    utils.plot_pnl(delta, df)

    n_sim = 300
    generator = GBM_Generator(r = r, sigma = sigma, S0 = S0, freq = freq)
    env_args["generator"] = generator
    env_args["testing"] = True
    pnl_paths_dict, pnl_dict, tcosts_dict, ntrades_dict = utils.simulate_pnl_PPO(model, delta_agent, n_sim, env_args)
    utils.plot_pnl_hist(pnl_paths_dict, pnl_dict, tcosts_dict, ntrades_dict)
