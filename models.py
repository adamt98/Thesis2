from Environments import DiscreteEnv
from Generators import GBM_Generator, HestonGenerator
import Utils

import random
import graphviz 
import numpy as np
from tqdm import tqdm as tq
from abc import ABC, abstractmethod

import torch as t
import torch.nn as nn
from torch.distributions import Categorical

from machin.frame.algorithms import DQN
from machin.frame.algorithms.dqn import DQN
from machin.frame.algorithms import PPO
from machin.frame.algorithms.ppo import PPO
from machin.utils.logging import default_logger as logger
from torch.utils.tensorboard import SummaryWriter

from sklearn.linear_model import LinearRegression
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.base import clone
from lineartree import LinearTreeRegressor

weights_dir = "./DQN_weights/"

# QNet model
class QNet(nn.Module):
    def __init__(self, state_dim, action_num, layers):
        super(QNet, self).__init__()
        self.n_layers = len(layers)
        bn_layers = [nn.LayerNorm(i, elementwise_affine=False) for i in layers]
        self.bn_layers = nn.ModuleList(bn_layers)
        layers = [state_dim] + layers + [action_num]
        fc_layers = [nn.Linear(first, second) for first, second in zip(layers, layers[1:])]
        self.fc_layers = nn.ModuleList(fc_layers)

    def forward(self, some_state):
        a = self.fc_layers[0](some_state)
        for i, fc in enumerate(self.fc_layers[1:]):
            a = t.relu(self.bn_layers[i](a))
            a = fc(a)
        return a

# PPO Actor model
class Actor(nn.Module):
    def __init__(self, state_dim, action_num, layers):
        super().__init__()
        self.n_layers = len(layers)
        bn_layers = [nn.LayerNorm(i, elementwise_affine=False) for i in layers]
        self.bn_layers = nn.ModuleList(bn_layers)
        layers = [state_dim] + layers + [action_num]
        fc_layers = [nn.Linear(first, second) for first, second in zip(layers, layers[1:])]
        self.fc_layers = nn.ModuleList(fc_layers)

    def forward(self, state, action=None):
        a = self.fc_layers[0](state)
        for i, fc in enumerate(self.fc_layers[1:]):
            a = t.relu(self.bn_layers[i](a))
            a = fc(a)

        probs = t.softmax(a, dim=1)
        dist = Categorical(probs=probs)
        act = action if action is not None else dist.sample()
        act_entropy = dist.entropy()
        act_log_prob = dist.log_prob(act.flatten())
        return act, act_log_prob, act_entropy

# PPO Critic model
class Critic(nn.Module):
    def __init__(self, state_dim, layers):
        super().__init__()

        self.n_layers = len(layers)
        bn_layers = [nn.LayerNorm(i, elementwise_affine=False) for i in layers]
        self.bn_layers = nn.ModuleList(bn_layers)
        layers = [state_dim] + layers + [1]
        fc_layers = [nn.Linear(first, second) for first, second in zip(layers, layers[1:])]
        self.fc_layers = nn.ModuleList(fc_layers)

    def forward(self, state):
        a = self.fc_layers[0](state)
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

# Delta Hedging benchmark
class DeltaHedge(Model):
    
    def __init__(self, K, call = True):
        self.K = K
        self.call = call
    
    def train(self):
        pass

    def predict_action(self, state, env : DiscreteEnv):
        _, spot, ttm = env.denormalize_state(state)
        delta = env.generator.get_delta(spot, self.K, ttm)
        shares = round(100*delta)
        if self.call: return shares 
        else: return shares - 100

    def test(self, env : DiscreteEnv):
        done = False
        state = env.reset()
        while not done:
            action = self.predict_action(state, env)
            state, _, terminal, info = env.step(action)
            done = terminal

        return info["output"]

class ModelAverager(Model):
    def __init__(self, env : DiscreteEnv, gamma):
        self.env = env
        self.type = LinearTreeRegressor(base_estimator= LinearRegression(), max_depth=19, min_samples_leaf=100)
        self.models = []
        self.gamma = gamma

    def reset(self):
        self.models = []

    def q_vals(self, X):
        if len(self.models) == 0:
            return np.full(X.shape[0], 4.0)
        else:
            #y = [model.predict(X) for model in self.models]
            return self.models[-1].predict(X) #np.mean(y, axis = 0)

    def consensus_q_vals(self, X):
        if len(self.models) == 0:
            return np.full(X.shape[0], 4.0)
        else:
            y = [model.predict(X) for model in self.models]
            return np.mean(y, axis = 0)

    # returns the best action according to consensus
    def predict_action(self, state, env : DiscreteEnv =None):
        if env is None:
            env = self.env

        appended = np.append(np.tile(state,(len(env.actions),1)), np.array(env.actions).reshape((-1,1)), axis=1)
        evals = self.q_vals(appended)
        # break ties randomly
        actionIndex = np.random.choice(np.flatnonzero(evals == np.max(evals)))
        return env.actions[actionIndex], evals[actionIndex]

    def predict_random(self, env : DiscreteEnv = None):
        if env is None: env = self.env
        return random.choice(env.actions)

    def train(self, n_steps, batches, eps_func):
        self.reset()
        for batch in range(batches):
            
            # roll out a single batch
            states = []
            actions = []
            rewards = []

            state = self.env.denormalize_state(self.env.reset())
            for i in tq(range(n_steps)):
                rnd = random.random()
                if rnd < eps_func(i) :
                    action = self.predict_random()
                else:
                    action, _ = self.predict_action(state)

                new_state, reward, terminal, _ = self.env.step(action)
                new_state = self.env.denormalize_state(new_state)

                actions.append(action)
                rewards.append(reward)
                states.append(state)

                if terminal:
                    state = self.env.denormalize_state(self.env.reset())
                else:
                    state = new_state

                # save the last state
                if i == (n_steps - 1):
                    states.append(state)
                    action, _ = self.predict_action(state)
                    actions.append(action)

            # build x,y data
            x = np.append(np.array(states).reshape((-1, 3)), np.array(actions).reshape((-1,1)), axis=1)
            q_vals = self.consensus_q_vals(x[1:]) # don't use the first state-action pair
            y = rewards + self.gamma * q_vals

            # train the last model
            self.models.append(clone(self.type))    
            self.models[-1].fit(x[:-1],y) # don't use the last state-action pair
            # if batch == batches -1:
            #     dot_data = tree.export_graphviz(fitted, out_file=None) 
            #     graph = graphviz.Source(dot_data) 
            #     graph.render("check") 
            #     dot_data = tree.export_graphviz(fitted, out_file=None, 
            #          feature_names=["holdings","price","ttm","action"],  
            #          class_names=["q val"],  
            #          filled=True, rounded=True,  
            #          special_characters=True)  
            #     graph = graphviz.Source(dot_data)  
            #     return graph 
            del x
            del q_vals
            del y

    def simulator_func(self, env : DiscreteEnv):
        done = False
        state = env.denormalize_state(env.reset())
        while not done:
            action, _ = self.predict_action(state, env)
            # print(state)
            # print(action)
            state, _, terminal, info = env.step(action)
            state = env.denormalize_state(state)
            done = terminal

        return info["output"]

    def test(self, generator : GBM_Generator, env_args, n_sim):
        
        test_env = DiscreteEnv(**env_args)
        df = self.simulator_func(test_env)
        
        test_env_delta = DiscreteEnv(**env_args)    
        delta_agent = DeltaHedge(generator.initial)
        delta = delta_agent.test(test_env_delta)

        Utils.plot_decisions(delta, df)

        Utils.plot_pnl(delta, df)

        generator = GBM_Generator(r = generator.r, sigma = generator.sigma, S0 = generator.initial, freq = generator.freq)
        env_args["generator"] = generator
        env_args["testing"] = True
        pnl_paths_dict, pnl_dict, tcosts_dict, ntrades_dict = Utils.simulate_pnl(delta_agent, n_sim, env_args, self.simulator_func)
        Utils.plot_pnl_hist(pnl_paths_dict, pnl_dict, tcosts_dict, ntrades_dict)

class DQN_Model(Model):

    def __init__(self,
                observe_dim,
                action_num,
                layers,
                epsilon_decay,
                learning_rate=1e-5,
                batch_size=32,
                discount=0.9,
                gradient_max=1.0,
                replay_size=750000,
                update_rate=1.0):
        
        self.observe_dim = observe_dim
        q_net = QNet(observe_dim, action_num, layers)
        q_net_target = QNet(observe_dim, action_num, layers)
        self.dqn = DQN(q_net, 
                        q_net_target,
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

    def train(self, max_episodes, env, solved_reward, solved_repeat, load_weights, save_weights):

        if load_weights:
            self.dqn.load(weights_dir, {"qnet_target" : "qnt"})

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

        if save_weights:
            self.dqn.save(weights_dir, {"qnet_target" : "qnt"}, version=0)

    def simulate(self, env):
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
                
        return info['output']

    # dont forget to pass a deep copy of the env_args & generator
    def test(self, generator : GBM_Generator, env_args, n_sim):
        env = DiscreteEnv(**env_args)
        df = self.simulate(env)

        # delta hedge benchmark
        test_env_delta = DiscreteEnv(**env_args)
        delta_agent = DeltaHedge(generator.initial)
        delta = delta_agent.test(test_env_delta)

        Utils.plot_decisions(delta, df)
        Utils.plot_pnl(delta, df)

        
        generator = GBM_Generator(r = generator.r, sigma = generator.sigma, S0 = generator.initial, freq = generator.freq)
        
        env_args["generator"] = generator
        env_args["testing"] = True
        pnl_paths_dict, pnl_dict, tcosts_dict, ntrades_dict = Utils.simulate_pnl(delta_agent, n_sim, env_args, self.simulate)
        Utils.plot_pnl_hist(pnl_paths_dict, pnl_dict, tcosts_dict, ntrades_dict)

class PPO_Model(Model):

    def __init__(self, 
                observe_dim, 
                action_num, 
                layers, 
                actor_learning_rate=1e-5,
                critic_learning_rate=1e-5,
                batch_size=32,
                discount=0.9,
                gradient_max=1.0,
                replay_size=750000,
                entropy_weight=None,
                value_weight=0.5,
                gae_lambda=1.0,
                surrogate_loss_clip=0.2):

        self.observe_dim = observe_dim
        self.actor = Actor(observe_dim, action_num, layers)
        self.critic = Critic(observe_dim, layers)

        self.model = PPO(self.actor, self.critic,
                        t.optim.Adam,
                        nn.MSELoss(reduction='sum'),
                        visualize=False,
                        actor_learning_rate=actor_learning_rate,
                        critic_learning_rate=critic_learning_rate,
                        batch_size=batch_size,
                        discount=discount,
                        gradient_max=gradient_max,
                        replay_size=replay_size,
                        entropy_weight=entropy_weight, 
                        value_weight=value_weight,
                        gae_lambda=gae_lambda,
                        surrogate_loss_clip = surrogate_loss_clip,
                        actor_update_times=50, # default 5
                        critic_update_times=100) # default 10

    def train(self, max_episodes, env, solved_reward, solved_repeat, load_weights, save_weights, writer : SummaryWriter):
        episode, step, reward_fulfilled = 0, 0, 0
        smoothed_total_reward = 0
        terminal = False

        if load_weights:
            self.model.load(weights_dir, 
                            {"restorable_model_1": "actor",
                            "restorable_model_2": "critic"})

        while episode < max_episodes:
            episode += 1
            total_reward = 0
            terminal = False
            step = 0
            state = t.tensor(env.reset(), dtype=t.float32).view(1, self.observe_dim)
            ep_list = []
            entropy = 0
            best_act_prob = 0
            while not terminal:
                step += 1
                with t.no_grad():
                    old_state = state
                    # agent model inference
                    action, act_log_prob, act_entropy = self.model.act(
                        {"state": old_state}
                    )
                    entropy += act_entropy
                    best_act_prob += np.exp(act_log_prob)
                    state, reward, terminal, _ = env.step(action.item())
                    state = t.tensor(state, dtype=t.float32).view(1, self.observe_dim)
                    total_reward += reward
                    
                    clipped_reward = np.clip(reward,-30.0,30.0)
                    ep_list.append({
                        "state": {"state": old_state},
                        "action": {"action": action},
                        "next_state": {"state": state},
                        "reward": clipped_reward,
                        "terminal": terminal
                    })

            self.model.store_episode(ep_list)
            mean_est_policy_val, val_loss  = self.model.update()
            
            writer.add_scalar("mean_prob_best_act",best_act_prob/step, episode)
            writer.add_scalar("mean_entropy", entropy/step, episode)
            writer.add_scalar("mean_est_policy_val", mean_est_policy_val, episode)
            writer.add_scalar("val_loss", val_loss, episode)
            # grad norm actor
            total_norm=0
            for p in self.actor.parameters():
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            writer.add_scalar("grad_norm_actor",total_norm,episode)

            # grad norm critic
            total_norm=0
            for p in self.critic.parameters():
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            writer.add_scalar("grad_norm_critic",total_norm,episode)


            

            # show reward
            smoothed_total_reward = (smoothed_total_reward * 0.9 +
                                        total_reward * 0.1)
            
            writer.add_scalar("smoothed_reward",smoothed_total_reward,episode)
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
            self.model.save(weights_dir, 
                            {"restorable_model_1": "actor",
                            "restorable_model_2": "critic"})

    def simulate(self, env):
        state = t.tensor(env.reset(), dtype=t.float32).view(1, self.observe_dim)
        terminal = False
        while not terminal:
            with t.no_grad():
                old_state = state
                # agent model inference
                
                action = self.model.act(
                    {"state": old_state},
                    use_target=True
                )[0]
                state, reward, terminal, info = env.step(action.item())
                state = t.tensor(state, dtype=t.float32).view(1, self.observe_dim)
                
        return info['output']

    def test(self, generator : GBM_Generator, env_args, n_sim):
        env = DiscreteEnv(**env_args)
        df = self.simulate(env)

        # delta hedge benchmark
        test_env_delta = DiscreteEnv(**env_args)
        delta_agent = DeltaHedge(generator.initial)
        delta = delta_agent.test(test_env_delta)

        Utils.plot_decisions(delta, df)
        Utils.plot_pnl(delta, df)

        generator = GBM_Generator(r = generator.r, sigma = generator.sigma, S0 = generator.initial, freq = generator.freq)
        env_args["generator"] = generator
        env_args["testing"] = True
        pnl_paths_dict, pnl_dict, tcosts_dict, ntrades_dict = Utils.simulate_pnl(delta_agent, n_sim, env_args, self.simulate)
        Utils.plot_pnl_hist(pnl_paths_dict, pnl_dict, tcosts_dict, ntrades_dict)