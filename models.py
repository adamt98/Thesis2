from Environments import DiscreteEnv
from Generators import GBM_Generator, HestonGenerator
import Utils

import random
import numpy as np
from tqdm import tqdm as tq
from abc import ABC, abstractmethod

import torch as t
import torch.nn as nn
from torch.distributions import Categorical

#from torch.utils.tensorboard import SummaryWriter


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

class PPOBase(nn.Module):
    def __init__(self, state_dim, action_num, layers):
        super().__init__()
        self.n_layers = len(layers)
        bn_layers = [nn.LayerNorm(i, elementwise_affine=True) for i in layers]
        self.bn_layers = nn.ModuleList(bn_layers)
        layers = [state_dim] + layers
        fc_layers = [nn.Linear(first, second) for first, second in zip(layers, layers[1:])]
        self.value_layer = nn.Linear(layers[-1],1)
        self.prob_layer = nn.Linear(layers[-1],action_num)
        self.fc_layers = nn.ModuleList(fc_layers)

    def forward_actor(self, state, action=None):
        a = self.fc_layers[0](state)
        a = t.relu(self.bn_layers[0](a))

        for i, fc in enumerate(self.fc_layers[1:]):
            a = fc(a)
            a = t.relu(self.bn_layers[i+1](a))

        a = self.prob_layer(a)
        probs = t.softmax(a, dim=1)
        dist = Categorical(probs=probs)
        act = action if action is not None else dist.sample()
        act_entropy = dist.entropy()
        act_log_prob = dist.log_prob(act.flatten())
        #print(dist.probs)
        return act, act_log_prob, act_entropy

    def forward_critic(self, state):
        a = self.fc_layers[0](state)
        a = t.relu(self.bn_layers[0](a))

        for i, fc in enumerate(self.fc_layers[1:]):
            a = fc(a)
            a = t.relu(self.bn_layers[i+1](a))

        a = self.value_layer(a)
        a = t.tanh(a)
        return a

# PPO Actor model
class Actor(nn.Module):
    def __init__(self, ppo_base : PPOBase):
        super().__init__()
        self.base = ppo_base

    def forward(self, state, action=None):
        return self.base.forward_actor(state, action)

# PPO Critic model
class Critic(nn.Module):
    def __init__(self, ppo_base : PPOBase):
        super().__init__()
        self.base = ppo_base

    def forward(self, state):
        return self.base.forward_critic(state)


# Delta Hedging benchmark
class DeltaHedge():
    
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

    def test(self, env : DiscreteEnv, state):
        done = False
        while not done:
            action = self.predict_action(state, env)
            state, _, terminal, info = env.step(action)
            done = terminal

        return info["output"]
