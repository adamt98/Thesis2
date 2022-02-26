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
import gym

observe_dim = 3
action_num = 101
max_episodes = 2000
solved_reward = -150
solved_repeat = 5


# model definition
class Net(nn.Module):
    def __init__(self, state_dim, action_num):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(state_dim, 16)
        self.bn1 = nn.LayerNorm(16)
        self.fc2 = nn.Linear(16, 16)
        self.bn2 = nn.LayerNorm(16)
        self.fc3 = nn.Linear(16, action_num)

    def forward(self, some_state):
        
        a = self.fc1(some_state)
        a = t.relu(self.bn1(a))
        a = self.fc2(a)
        a = t.relu(self.bn2(a))
        return self.fc3(a)

S0 = 100

# Annualized
sigma = 0.01*np.sqrt(250) # 1% vol per day
r = 0.0

freq = 0.2 # corresponds to trading freq of 5x per day
ttm = 50
kappa = 0.1
cost_multiplier = 0.0
gamma = 0.99

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

actor = Net(observe_dim, action_num)
critic = Net(observe_dim, action_num)

model = PPO(actor,critic,
                t.optim.Adam,
                nn.MSELoss(reduction='sum'),
                visualize=False)

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
                    {"some_state": old_state}
                )
                state, reward, terminal, _ = env.step(action.item())
                state = t.tensor(state, dtype=t.float32).view(1, observe_dim)
                total_reward += reward
                
                ep_list.append({
                    "state": {"some_state": old_state},
                    "action": {"action": action},
                    "next_state": {"some_state": state},
                    "reward": reward,
                    "terminal": terminal
                })

        model.store_episode(ep_list)
        # update, update more if episode is longer, else less
        if episode > 100:
            for _ in range(step):
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
            action = model.act_discrete(
                {"some_state": old_state}
            )
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
    pnl_paths_dict, pnl_dict, tcosts_dict, ntrades_dict = utils.simulate_pnl_DQN(model, delta_agent, n_sim, env_args)
    utils.plot_pnl_hist(pnl_paths_dict, pnl_dict, tcosts_dict, ntrades_dict)
