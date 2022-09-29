from Environments import DiscreteEnv
from Environments import BarrierEnv, BarrierEnv2, BarrierEnv3#, BarrierEnv4
from tqdm import tqdm
import numpy as np

# Delta Hedging benchmark
class DeltaHedge():
    
    def __init__(self, K, call = True, n_puts_sold = None, min_action = 0, put_K = None):
        self.K = K
        self.call = call
        self.n_puts_sold = n_puts_sold
        self.min_action = min_action
        self.put_K = put_K
    
    def predict_action(self, state, env : DiscreteEnv | BarrierEnv | BarrierEnv2 | BarrierEnv3):
        if type(env) == BarrierEnv2:
            _, spot, ttm, _ = env.denormalize_state(state)
        # elif type(env) == BarrierEnv4:
        #     _, spot, _, ttm = env.denormalize_state(state)
        elif type(env) == BarrierEnv3:
            _, spot, ttm, _, _ = env.denormalize_state(state)
        else:
            _, spot, ttm = env.denormalize_state(state)

        if type(env) == DiscreteEnv:
            delta = env.generator.get_delta(spot, self.K, ttm, call=self.call)
            shares = round(100*delta)
            return shares

        elif type(env) == BarrierEnv3:
            delta_DIP = env.generator.get_DIP_delta(spot, self.K, ttm)
            delta_put = env.generator.get_delta(spot, self.put_K, ttm)
            delta_portfolio = delta_DIP - self.n_puts_sold*delta_put
            shares = - round(100*delta_portfolio)
            return [shares - self.min_action, 1] # due to space.Discrete being stupid

        else:
            delta_DIP = env.generator.get_DIP_delta(spot, self.K, ttm)
            delta_put = env.generator.get_delta(spot, self.put_K, ttm)
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


# Re-hedging done only when accumulated delta exceeds a threshold, threshold found via mean-variance optimization
class ThresholdDelta():
    def __init__(self, K, call = True, n_puts_sold = None, min_action = 0, put_K = None):
        self.K = K
        self.call = call
        self.n_puts_sold = n_puts_sold
        self.min_action = min_action
        self.put_K = put_K
        self.optimalThreshold = None

    def train(self, env, episodes, kappa):
        thresholds = [0,25,50,75,100]#,200]
        means = []
        variances = []
        for threshold in thresholds:
            finalPnL = []
            for i in tqdm(range(episodes)):
                obs = env.reset_with_seed(i*1301)
                df = self.test(env, obs, threshold)
                finalPnL.append(df.pnl.cumsum().values[-1])

            means.append(np.mean(finalPnL))
            variances.append(np.var(finalPnL))

        # pick the best threshold
        scores = [mean - 0.5*kappa*var for mean,var in zip(means,variances)]
        self.optimalThreshold = thresholds[np.argmax(scores)]
        print(means)
        print("____")
        print(variances)
        print("Saving optimal threshold = ", self.optimalThreshold)

            
    def predict_action(self, state, env : DiscreteEnv | BarrierEnv | BarrierEnv2 | BarrierEnv3, threshold):
        if (self.optimalThreshold is None) and (threshold is None):
            print("WARNING: Threshold not optimized. Using threshold = 0.")
            threshold : int = 0
        elif threshold is None: 
            threshold : int = self.optimalThreshold
        else:
            threshold : int = threshold

        if type(env) == BarrierEnv2:
            holdings, spot, ttm, _ = env.denormalize_state(state)
        elif type(env) == BarrierEnv3:
            holdings, spot, ttm, _, _ = env.denormalize_state(state)
        else:
            holdings, spot, ttm = env.denormalize_state(state)

        if type(env) == DiscreteEnv:
            delta = env.generator.get_delta(spot, self.K, ttm, call = self.call)
            shares = round(100*delta)
            if np.abs(shares - holdings) < int(threshold): return holdings

            return shares

        elif type(env) == BarrierEnv3:
            delta_DIP = env.generator.get_DIP_delta(spot, self.K, ttm)
            delta_put = env.generator.get_delta(spot, self.put_K, ttm)
            delta_portfolio = delta_DIP - self.n_puts_sold*delta_put
            shares = - round(100*delta_portfolio)

            if np.abs(shares - holdings) < int(threshold): 
                return [holdings - self.min_action, 1]
            else: 
                return [shares - self.min_action, 1] # due to space.Discrete being stupid

        else:
            delta_DIP = env.generator.get_DIP_delta(spot, self.K, ttm)
            delta_put = env.generator.get_delta(spot, self.put_K, ttm)
            delta_portfolio = delta_DIP - self.n_puts_sold*delta_put
            shares = - round(100*delta_portfolio)

            if np.abs(shares - holdings) < int(threshold): 
                return holdings - self.min_action
            else: 
                return shares - self.min_action # due to space.Discrete being stupid

    def test(self, env, state, threshold = None):
        done = False
        while not done:
            action = self.predict_action(state, env, threshold)
            state, _, terminal, info = env.step(action)
            done = terminal

        return info["output"]
