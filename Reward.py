import numpy as np

class Reward():
    """Provides reward functions for the environment

    Types
    ----------
    "basic" : reward func implemented by all the research papers
    "static": takes into account the static hedge value

    Methods
    -------
    get_reward_func(type)
        gets the specified reward function type

    """
    def __init__(self, kappa):
        self.kappa = kappa

    def _basic_reward(self, new_opt_val, old_opt_val, new_und, old_und, trading_cost, holdings):
        pnl = - 100 * (new_opt_val - old_opt_val) + (new_und - old_und) * holdings
        return (( pnl - 0.5 * self.kappa * (pnl**2) ) - trading_cost) / 100.0

    def _dummy_reward(self, new_opt_val, old_opt_val, new_und, old_und, trading_cost, holdings):
        return holdings-50

    # we are long a single DIP
    def _static_hedge_reward(self, new_barrier_val, old_barrier_val, new_static_hedge_val, old_static_hedge_val, new_und, old_und, trading_cost, holdings):
        pnl = 100 * (new_barrier_val - old_barrier_val) - 100 * (new_static_hedge_val - old_static_hedge_val) + (new_und - old_und) * holdings
        rew = ( pnl - 0.5 * self.kappa * (pnl**2) - trading_cost) / 100.0
        return np.clip(rew, -3, 10.0), pnl - trading_cost

    # we are long a single DIP
    def _dynamic_hedge_reward(self, new_barrier_val, old_barrier_val, opt_val_change, opt_holdings, new_und, old_und, trading_cost, holdings):
        pnl = 100 * (new_barrier_val - old_barrier_val) - 100 * opt_holdings * opt_val_change + (new_und - old_und) * holdings
        rew = ( pnl - 0.5 * self.kappa * (pnl**2) - trading_cost) / 100.0
        return np.clip(rew, -3, 10.0), pnl - trading_cost

    def get_reward_func(self, type):
        if type == "basic":
            return self._basic_reward
        elif type == "dummy":
            return self._dummy_reward
        elif type == "static":
            return self._static_hedge_reward
        elif type == "dynamic":
            return self._dynamic_hedge_reward
        else:
            raise "Unknown reward func type {type}"