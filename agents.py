from numpy.core.fromnumeric import shape
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.base import clone
from discrete_environments import DiscreteEnv
from scipy.stats import norm
from tqdm import tqdm as tq
import random
import graphviz 
from sklearn import tree
from lineartree import LinearTreeRegressor

# RL models from stable-baselines
from stable_baselines3 import A2C, PPO, TD3, DDPG, SAC, DQN
import config

MODELS = {"a2c": A2C, "ddpg": DDPG, "td3": TD3, "sac": SAC, "ppo": PPO, "dqn": DQN}
MODEL_KWARGS = {x: config.__dict__[f"{x.upper()}_PARAMS"] for x in MODELS.keys()}
TENSORBOARD_LOG_DIR = f"tensorboard_log"


class ModelAverager():
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

            state = self.env.reset()
            for i in tq(range(n_steps)):
                # print info
                # print("State: ", state)
                # print("Option value: ", self.env.generator.get_option_value(100, state[-1]))
                # predict action
                rnd = random.random()
                if rnd < eps_func(i) :
                    action = self.predict_random()
                    # print("random action: ", action)
                else:
                    action, _ = self.predict_action(state)
                    # print("greedy action: ", action)

                new_state, reward, terminal, _ = self.env.step(action)
                # print("Reward = %d", reward)
                # print("New state: ", new_state)
                # print("New option value: ", self.env.generator.get_option_value(100, new_state[-1]))
                # print("______________________")
                actions.append(action)
                rewards.append(reward)
                states.append(state)

                if terminal:
                    state = self.env.reset()
                else:
                    state = new_state

                # save the last state
                if i == (n_steps - 1):
                    states.append(state)
                    action, _ = self.predict_action(state)
                    actions.append(action)

            # build x,y data
            x = np.append(np.array(states).reshape((-1, 3)), np.array(actions).reshape((-1,1)), axis=1)#[np.append(state, action) for state, action in zip(states, actions)]
            q_vals = self.consensus_q_vals(x[1:]) # don't use the first state-action pair
            y = rewards + self.gamma * q_vals

            # train the last model
            self.models.append(clone(self.type))    
            self.models[-1].fit(x[:-1],y) # don't use the last state-action pair
            del x
            del q_vals
            del y
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

    def test(self, env : DiscreteEnv):
        done = False
        state = env.reset()
        while not done:
            action, _ = self.predict_action(state, env)
            # print(state)
            # print(action)
            state, _, terminal, info = env.step(action)
            done = terminal

        return info["output"]

class DeltaHedge():
    
    def __init__(self, r, sigma, K, call = True):
        self.r = r
        self.sigma = sigma
        self.K = K
        self.call = call
    
    def predict_action(self, state, env : DiscreteEnv):
        holdings, spot, ttm = state[0], state[1], state[2]
        
        delta = env.generator.get_delta(spot, self.K, ttm)
        
        # compute actions
        if self.call:
            #return round(np.clip(100*delta - holdings, a_min = env.actions[0], a_max = env.actions[-1]))
            return round(100*delta)
        else:
            return round(100*delta - 100)

    def test(self, env : DiscreteEnv):
        done = False
        state = env.reset()
        while not done:
            action = self.predict_action(state, env)
            state, _, terminal, info = env.step(action)
            done = terminal

        return info["output"]

class DRLAgent():
    @staticmethod
    def Prediction(model, environment, pred_args={}):
        test_env, test_obs = environment.get_sb_env()
        """make a prediction"""
        
        for i in range(environment.expiry):
            action, _states = model.predict(test_obs, **pred_args)
            test_obs, rewards, dones, info = test_env.step(action)
            if dones[0]:
                print("hit end!")
                break
                
        return info[0]
    
    def __init__(self, env):
        self.env = env

    def train_model(self, model, tb_log_name, total_timesteps=5000, n_eval_episodes=5 ):
        model = model.learn(total_timesteps=total_timesteps, tb_log_name=tb_log_name, n_eval_episodes=n_eval_episodes)
        return model
        
    def get_model(
        self,
        model_name,
        policy="MlpPolicy",
        policy_kwargs=None,
        model_kwargs=None,
        verbose=1,
    ):
        if model_name not in MODELS:
            raise NotImplementedError("NotImplementedError")

        if model_kwargs is None:
            model_kwargs = MODEL_KWARGS[model_name]

        print(model_kwargs)
        model = MODELS[model_name](
            policy=policy,
            env=self.env,
            tensorboard_log=f"{TENSORBOARD_LOG_DIR}/{model_name}",
            verbose=verbose,
            policy_kwargs=policy_kwargs,
            **model_kwargs,
        )
        return model