from Environments import DiscreteEnv
from Environments2 import BarrierEnv, BarrierEnv2, BarrierEnv3, BarrierEnv4
from Generators import GBM_Generator, HestonGenerator
import Utils
import numpy as np


# Delta Hedging benchmark
class DeltaHedge():
    
    def __init__(self, K, call = True, n_puts_sold = None, min_action = 0, put_K = None):
        self.K = K
        self.call = call
        self.n_puts_sold = n_puts_sold
        self.min_action = min_action
        self.put_K = put_K
    
    def train(self):
        pass

    def predict_action(self, state, env : DiscreteEnv | BarrierEnv | BarrierEnv2):
        if type(env) == BarrierEnv2:
            _, spot, ttm, _ = env.denormalize_state(state)
        elif type(env) == BarrierEnv4:
            _, spot, _, ttm = env.denormalize_state(state)
        elif type(env) == BarrierEnv3:
            _, spot, ttm, _, _ = env.denormalize_state(state)
        else:
            _, spot, ttm = env.denormalize_state(state)

        if type(env) == DiscreteEnv:

            delta = env.generator.get_delta(spot, self.K, ttm)
            shares = round(100*delta)
            if self.call: return shares 
            else: return shares - 100

        elif type(env) == BarrierEnv3:
            # vega_DIP = env.generator.get_DIP_vega(spot, self.K, ttm)
            # vanilla_DIP = env.generator.get_vega(spot, self.put_K, ttm)
            # n_puts = round(vega_DIP/vanilla_DIP)

            # delta_DIP = env.generator.get_DIP_delta(spot, self.K, ttm)
            # delta_put = env.generator.get_delta(spot, self.put_K, ttm) - 1
            # delta_portfolio = delta_DIP - n_puts*delta_put
            # shares = - round(100*delta_portfolio)
            delta_DIP = env.generator.get_DIP_delta(spot, self.K, ttm)
            delta_put = env.generator.get_delta(spot, self.put_K, ttm) - 1
            delta_portfolio = delta_DIP - self.n_puts_sold*delta_put
            shares = - round(100*delta_portfolio)
            return [shares - self.min_action, 1] # due to space.Discrete being stupid

        else:
            delta_DIP = env.generator.get_DIP_delta(spot, self.K, ttm)
            delta_put = env.generator.get_delta(spot, self.put_K, ttm) - 1
            delta_portfolio = delta_DIP - self.n_puts_sold*delta_put
            shares = - round(100*delta_portfolio)
            return shares - self.min_action # due to space.Discrete being stupid

    def test(self, env : DiscreteEnv | BarrierEnv, state):
        done = False
        while not done:
            action = self.predict_action(state, env)
            state, _, terminal, info = env.step(action)
            done = terminal

        return info["output"]
