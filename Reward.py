class Reward():
    """Provides reward functions for the environment

    Types
    ----------
    "basic" : reward func implemented by all the research papers

    Methods
    -------
    get_reward_func(type)
        gets the specified reward function type

    """
    def __init__(self, kappa):
        self.kappa = kappa

    def _basic_reward(self, new_opt_val, old_opt_val, new_und, old_und, trading_cost, holdings):
        pnl = - 100 * (new_opt_val - old_opt_val) + (new_und - old_und) * holdings
        return ( pnl - 0.5 * self.kappa * (pnl**2) ) - trading_cost

    def get_reward_func(self, type):
        if type == "basic":
            return self._basic_reward
        else:
            raise "Unknown reward func type {type}"