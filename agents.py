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

class ModelAverager():
    def __init__(self, env : DiscreteEnv, gamma):
        self.env = env
        self.type = DecisionTreeRegressor()
        self.models = []
        self.gamma = gamma

    def reset(self):
        self.models = []

    def q_vals(self, X):
        X = np.array(X).reshape((len(X),len(X[0])))
        if len(self.models) == 0:
            return np.full(X.shape[0], 5.0)
        else:
            y = [model.predict(X) for model in self.models]
            return np.mean(y, axis = 0)

    # returns the best action according to consensus
    def predict_action(self, state, env : DiscreteEnv =None):
        if env is None:
            env = self.env

        evals = [self.q_vals([np.append(state, action)]) for action in env.actions]
        # break ties randomly
        actionIndex = np.random.choice(np.flatnonzero(evals == np.max(evals)))
        return env.actions[actionIndex], evals[actionIndex]

    def predict_random(self, env : DiscreteEnv = None):
        if env is None:
            env = self.env
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
            x = [np.append(state, action) for state, action in zip(states, actions)]
            q_vals = self.q_vals(x[1:]) # don't use the first state-action pair
            y = rewards + self.gamma * q_vals

            # train the last model
            self.models.append(clone(self.type))    
            self.models[-1].fit(x[:-1],y) # don't use the last state-action pair
            del x
            del q_vals
            del y

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