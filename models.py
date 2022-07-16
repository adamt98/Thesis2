from Environments import DiscreteEnv
from Environments2 import BarrierEnv
from Generators import GBM_Generator, HestonGenerator
import Utils


# Delta Hedging benchmark
class DeltaHedge():
    
    def __init__(self, K, call = True, n_puts_sold = None, min_action = 0):
        self.K = K
        self.call = call
        self.n_puts_sold = n_puts_sold
        self.min_action = min_action
    
    def train(self):
        pass

    def predict_action(self, state, env : DiscreteEnv | BarrierEnv):
        _, spot, ttm = env.denormalize_state(state)

        if type(env) == DiscreteEnv:

            delta = env.generator.get_delta(spot, self.K, ttm)
            shares = round(100*delta)
            if self.call: return shares 
            else: return shares - 100
        else:
            delta_DIP = env.generator.get_DIP_delta(spot, self.K, ttm)
            delta_put = env.generator.get_delta(spot, self.K, ttm) - 1
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
