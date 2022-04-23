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

from agents import ModelAverager, DeltaHedge, DRLAgent
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

from abc import ABC, abstractmethod

# QNet model definition
class QNet(nn.Module):
    def __init__(self, state_dim, action_num, layers):
        super(QNet, self).__init__()
        self.n_layers = len(layers)
        self.bn_layers = [nn.LayerNorm(i, elementwise_affine=False) for i in layers]
        layers = [state_dim] + layers + [action_num]
        self.fc_layers = [nn.Linear(first, second) for first, second in zip(layers, layers[1:])]

    def forward(self, some_state):
        a = self.fc_layers[0](some_state)
        for i, fc in enumerate(self.fc_layers[1:]):
            a = t.relu(self.bn_layers[i](a))
            a = fc(a)
        return a

# Abstract model class
class Model(ABC):
 
    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def test(self):
        pass

class DQN_Model(Model):

    def __init__(self,
                observe_dim,
                action_num,
                epsilon_decay,
                learning_rate=1e-5,
                batch_size=32,
                discount=0.9,
                gradient_max=1.0,
                replay_size=750000,
                update_rate=1.0):
        
        self.observe_dim = observe_dim
        q_net = QNet(observe_dim, action_num)
        q_net_target = QNet(observe_dim, action_num)
        self.dqn = DQN(q_net, q_net_target,
                    t.optim.Adam,
                    nn.MSELoss(reduction='sum'),
                    visualize=False,
                    learning_rate=learning_rate,
                    batch_size=batch_size,
                    discount=discount,
                    gradient_max=gradient_max,
                    epsilon_decay=epsilon_decay,
                    replay_size=replay_size,
                    update_rate=update_rate,
                    mode="fixed_target"
                    )
        return 9

    def train(self, max_episodes, env, solved_reward, solved_repeat):
        episode, step, reward_fulfilled = 0, 0, 0
        smoothed_total_reward = 0
        terminal = False
        eps = 1.0
        while episode < max_episodes:
            episode += 1
            total_reward = 0
            terminal = False
            step = 0

            state = env.reset()
            state = t.tensor(state, dtype=t.float32).view(1, self.observe_dim)
            ep_list = []
            while not terminal:
                step += 1
                with t.no_grad():
                    old_state = state
                    # agent model inference
                    if episode % 2 == 0 : decay_eps = True 
                    else: decay_eps = False
                    action = self.dqn.act_discrete_with_noise(
                        {"some_state": old_state},
                        use_target=True,
                        decay_epsilon=True
                    )
                    eps = eps*self.dqn.epsilon_decay
                    state, reward, terminal, _ = env.step(action.item())
                    
                    total_reward += reward

                    clipped_reward = np.clip(reward,-10.0,10.0)

                    state = t.tensor(state, dtype=t.float32).view(1, self.observe_dim)
                    ep_list.append({
                        "state": {"some_state": old_state},
                        "action": {"action": action},
                        "next_state": {"some_state": state},
                        "reward": clipped_reward,
                        "terminal": terminal
                    })

            self.dqn.store_episode(ep_list)
            # update target net
            if (episode > 500):
                for _ in range(50):
                    self.dqn.update(update_value=True,update_target=False)

            # one update of target net
            if (episode % 1000 == 0):
                self.dqn.update(update_value=False,update_target=True)

            # show reward
            smoothed_total_reward = total_reward#(smoothed_total_reward * 0.9 +
                                        #total_reward * 0.1)
            logger.info("Episode {} total reward={:.2f}"
                        .format(episode, smoothed_total_reward))

            if smoothed_total_reward > solved_reward:
                reward_fulfilled += 1
                if reward_fulfilled >= solved_repeat:
                    logger.info("Environment solved!")
                    break
            else:
                reward_fulfilled = 0

    def simulate(self):
        state = t.tensor(env.reset(), dtype=t.float32).view(1, self.observe_dim)
        terminal = False
        while not terminal:
            with t.no_grad():
                old_state = state
                # agent model inference
                action = model.act_discrete(
                    {"some_state": old_state},
                    use_target=True
                )
                state, reward, terminal, info = env.step(action.item())
                state = t.tensor(state, dtype=t.float32).view(1, self.observe_dim)
                
        return info['output']

    def test(self):
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
        state = t.tensor(env.reset(), dtype=t.float32).view(1, self.observe_dim)
        terminal = False
        while not terminal:
            with t.no_grad():
                old_state = state
                # agent model inference
                action = self.dqn.act_discrete(
                    {"some_state": old_state},
                    use_target=True
                )
                state, reward, terminal, info = env.step(action.item())
                state = t.tensor(state, dtype=t.float32).view(1, self.observe_dim)
                
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
        pnl_paths_dict, pnl_dict, tcosts_dict, ntrades_dict = utils.simulate_pnl_DQN(self.dqn, delta_agent, n_sim, env_args)
        utils.plot_pnl_hist(pnl_paths_dict, pnl_dict, tcosts_dict, ntrades_dict)

    