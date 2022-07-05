from sklearn.model_selection import learning_curve
from Environments import DiscreteEnv
from Generators import GBM_Generator
import Models
import Utils
import numpy as np
import matplotlib.pyplot as plt


from machin.utils.tensor_board import TensorBoard

import torch
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

#board = TensorBoard()
#board.init()
#print(board.is_inited())
# for i in range(100):
#     writer.add_scalar("ABC", 0.2*i, i)
    
# writer.close()
##### Environment config ###############

sigma = 0.01*np.sqrt(250) # 1% vol per day, annualized
r = 0.0 # Annualized
S0 = 100
freq = 0.2 # corresponds to trading freq of 5x per day
ttm = 50
kappa = 0.15
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
##### Training hyperparameter setup ######

observe_dim = 3
action_num = 101
solved_reward = 100
solved_repeat = 7
max_episodes = 35000

# 1 epoch = 3000 episodes = 150k time-steps
epoch = 150000
batch_size = 32 
n_epochs_per_update = 5

#n_updates = int(epoch * n_epochs_per_update / batch_size)
final_eps = 0.05
eps_decay = np.exp(np.log(final_eps)/(max_episodes*50))

layers = [20, 20, 20, 20]
learning_rate = 1e-5

n_sim = 100

## PPO setup
gae_lambda = 0.95
value_weight = - 0.2
entropy_weight = 0.5
actor_lr = 1e-4
critic_lr = 1e-4
surrogate_loss_clip = 0.1 # min and max acceptable KL divergence

## Model Averager setup
n_steps = 500
n_batches = 3
eps_func = Utils.EpsFunction(n_steps).get_func()

if __name__ == "__main__":
    ## TESTING DQN
    dqn = Models.DQN_Model(observe_dim, action_num, layers, eps_decay, learning_rate=learning_rate, batch_size=batch_size, discount=discount)
    dqn.train(max_episodes=max_episodes, env=env, solved_reward=solved_reward, solved_repeat=solved_repeat, load_weights=False, save_weights=True)

    generator = GBM_Generator(S0, r, sigma, freq)
    env_args = {
        "generator" : generator,
        "ttm" : ttm,
        "kappa" : kappa,
        "cost_multiplier" : cost_multiplier,
        "reward_type" : "basic",
        "testing" : True
    }
    dqn.test(generator=generator, env_args=env_args, n_sim=n_sim)

    ## TESTING PPO
    # ppo = Models.PPO_Model(observe_dim, action_num, layers, batch_size=batch_size, discount=discount,surrogate_loss_clip=surrogate_loss_clip,
    #                         gae_lambda=gae_lambda, entropy_weight=entropy_weight, value_weight=value_weight, actor_learning_rate=actor_lr, critic_learning_rate=critic_lr)
    # ppo.train(max_episodes=max_episodes, env=env, solved_reward=solved_reward, solved_repeat=solved_repeat, load_weights=False, save_weights=True, writer=writer)

    # generator = GBM_Generator(S0, r, sigma, freq)
    # env_args = {
    #     "generator" : generator,
    #     "ttm" : ttm,
    #     "kappa" : kappa,
    #     "cost_multiplier" : cost_multiplier,
    #     "reward_type" : "basic",
    #     "testing" : True
    # }

    #ppo.test(generator=generator, env_args=env_args, n_sim=n_sim)

    ## TESTING MODEL AVERAGER
    # agent = Models.ModelAverager(env, discount)
    # agent.train(n_steps, n_batches, eps_func)

    # generator = GBM_Generator(S0, r, sigma, freq)
    # env_args = {
    #     "generator" : generator,
    #     "ttm" : ttm,
    #     "kappa" : kappa,
    #     "cost_multiplier" : cost_multiplier,
    #     "reward_type" : "basic",
    #     "testing" : True
    # }
    # agent.test(generator, env_args, n_sim=n_sim)

    # gen = GBM_Generator(100.0, 0.0, 0.2, 1, None, 80.0)
    # und = []
    # opt = []
    # bar = []
    
    # for i in range(50):
    #     #und.append(gen.current)
    #     bar.append(gen.get_DIP_vega(spot = 60.0 + i, K = 100.0, ttm = 5))
    #     opt.append(gen.get_vega(spot = 60.0 + i, K = 80.0, ttm = 5) )
    #     #gen.get_next()

    # plt.figure(1, figsize=(12, 8))
    # # plt.subplot(121)
    # # plt.plot(und)
    # # plt.subplot(122)
    # plt.plot(opt)
    # plt.plot(bar)
    # plt.show()