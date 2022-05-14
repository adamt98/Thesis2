from sklearn.model_selection import learning_curve
from Environments import DiscreteEnv
from Generators import GBM_Generator
import Models
import Utils
import numpy as np


##### Environment config ###############

sigma = 0.01*np.sqrt(250) # 1% vol per day, annualized
r = 0.0 # Annualized
S0 = 100
freq = 0.2 # corresponds to trading freq of 5x per day
ttm = 50
kappa = 0.1
cost_multiplier = 0.0
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
##### Training hyperparameter setup ######

observe_dim = 3
action_num = 101
solved_reward = 0
solved_repeat = 7
max_episodes = 5000

# 1 epoch = 3000 episodes = 150k time-steps
epoch = 150000
batch_size = 32
n_epochs_per_update = 5

#n_updates = int(epoch * n_epochs_per_update / batch_size)
final_eps = 0.05
eps_decay = np.exp(np.log(final_eps)/(max_episodes*50))

layers = [16, 32, 32, 64]
learning_rate = 1e-5

n_sim = 300

## Model Averager setup
n_steps = 500
n_batches = 3
eps_func = Utils.EpsFunction(n_steps).get_func()

if __name__ == "__main__":
    ## TESTING DQN
    # dqn = Models.DQN_Model(observe_dim, action_num, layers, eps_decay, learning_rate=learning_rate, batch_size=batch_size, discount=discount)
    # dqn.train(max_episodes=max_episodes, env=env, solved_reward=solved_reward, solved_repeat=solved_repeat, load_weights=False, save_weights=False)

    # generator = GBM_Generator(S0, r, sigma, freq)
    # env_args = {
    #     "generator" : generator,
    #     "ttm" : ttm,
    #     "kappa" : kappa,
    #     "cost_multiplier" : cost_multiplier,
    #     "reward_type" : "basic",
    #     "testing" : True
    # }
    # dqn.test(generator=generator, env_args=env_args, n_sim=n_sim)

    ## TESTING PPO
    ppo = Models.PPO_Model(observe_dim, action_num, layers, learning_rate=learning_rate, batch_size=batch_size, discount=discount)
    ppo.train(max_episodes=max_episodes, env=env, solved_reward=solved_reward, solved_repeat=solved_repeat, load_weights=False, save_weights=True)

    generator = GBM_Generator(S0, r, sigma, freq)
    env_args = {
        "generator" : generator,
        "ttm" : ttm,
        "kappa" : kappa,
        "cost_multiplier" : cost_multiplier,
        "reward_type" : "basic",
        "testing" : True
    }

    ppo.test(generator=generator, env_args=env_args, n_sim=n_sim)

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