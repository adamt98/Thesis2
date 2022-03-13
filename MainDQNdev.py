from agents import DeltaHedge
from discrete_environments import DiscreteEnv, DiscreteEnv2
from data_generators import GBM_Generator
import utils
import numpy as np
import matplotlib.pyplot as plt

from machin.frame.algorithms import DQN
from machin.frame.algorithms.dqn import DQN
from machin.utils.logging import default_logger as logger
import torch as t
import torch.nn as nn

observe_dim = 3
action_num = 101
max_episodes = 10000
solved_reward = 0
solved_repeat = 7

load_weights = False
save_weights = False
weights_dir = "./DQN_weights/"

# model definition
class QNet(nn.Module):
    def __init__(self, state_dim, action_num):
        super(QNet, self).__init__()

        self.fc1 = nn.Linear(state_dim, 10)
        self.bn1 = nn.LayerNorm(10, elementwise_affine=False)
        self.fc2 = nn.Linear(10, 10)
        self.bn2 = nn.LayerNorm(10, elementwise_affine=False)
        self.fc3 = nn.Linear(10, 10)
        self.bn3 = nn.LayerNorm(10, elementwise_affine=False)
        self.fc4 = nn.Linear(10, action_num)

    def forward(self, some_state):
        a = self.fc1(some_state)
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
gamma = 0.88

generator = GBM_Generator(S0, r, sigma, freq)

env_args = {
    "generator" : generator,
    "ttm" : ttm,
    "kappa" : kappa,
    "cost_multiplier" : cost_multiplier,
    "testing" : False
}

env = DiscreteEnv2(**env_args)

# 1 epoch = 3000 episodes = 150k time-steps
#epoch = 150000
#n_epochs_per_update = 5
#n_updates = int(epoch * n_epochs_per_update / batch_size)

batch_size = 32
final_eps = 0.05
eps_decay = np.exp(np.log(final_eps)/(max_episodes*ttm))


q_net = QNet(observe_dim, action_num)
q_net_t = QNet(observe_dim, action_num)
dqn = DQN(q_net, q_net_t,
            t.optim.Adam,
            nn.MSELoss(reduction='sum'),
            visualize=False,
            learning_rate=1e-5,
            batch_size=batch_size,
            discount=gamma,
            gradient_max=1.0,
            epsilon_decay=eps_decay,
            replay_size=750000,
            update_rate=1.0,
            mode="fixed_target"
            )

if load_weights:
    dqn.load(weights_dir, {"qnet_target" : "qnt"})

episode, step, reward_fulfilled = 0, 0, 0
smoothed_total_reward = 0
terminal = False
eps = 1.0

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
                action = dqn.act_discrete_with_noise(
                    {"some_state": old_state},
                    use_target=True,
                    decay_epsilon=True
                )
                #eps = eps*dqn.epsilon_decay
                state, reward, terminal, _ = env.step(action.item())
                state = t.tensor(state, dtype=t.float32).view(1, observe_dim)
                total_reward += reward
                clipped_reward = np.clip(reward,-10.0,10.0)

                ep_list.append({
                    "state": {"some_state": old_state},
                    "action": {"action": action},
                    "next_state": {"some_state": state},
                    "reward": clipped_reward,
                    "terminal": terminal
                })

        dqn.store_episode(ep_list)
        # update target net
        if (episode > 500):
            for _ in range(50):
                dqn.update(update_value=True,update_target=False)

        # one update of target net
        if (episode % 1000 == 0):
            dqn.update(update_value=False,update_target=True)

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

    if save_weights:
        dqn.save(weights_dir, {"qnet_target" : "qnt"}, version=0)

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
            action = dqn.act_discrete(
                {"some_state": old_state},
                use_target=True
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
    pnl_paths_dict, pnl_dict, tcosts_dict, ntrades_dict = utils.simulate_pnl_DQN(dqn, delta_agent, n_sim, env_args)
    utils.plot_pnl_hist(pnl_paths_dict, pnl_dict, tcosts_dict, ntrades_dict)