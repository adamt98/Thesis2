import numpy as np
import pandas as pd
from data_generators import GBM_Generator

class DiscreteEnv():

    def __init__(self,
                generator : GBM_Generator,
                ttm,
                kappa = 0.1,
                testing = False):
        
        # Init some constants
        self.expiry = ttm
        self.kappa = kappa
        self.terminal = False     
        self.testing = testing
        self.cost = 0
        self.trades = 0
        self.episode = 0
        self.K = generator.current # ATM

        self.generator = generator
        
        # Create discrete action space
        self.actions = np.arange(0, 101)

        # Memorize some things for plotting purposes
        self.option_memory = []
        self.underlying_memory = []
        self.cost_memory = []
        self.trades_memory = []
        self.holdings_memory = []
        self.rewards_memory = []
        self.actions_memory = []
        self.pnl_memory = []
        
        # Initalize state
        self.state = self._initiate_state()

    def _initiate_state(self):
        return [0, self.generator.current, self.expiry]

    def step(self, action):
        self.terminal = (self.state[-1] == 1)
        
        # calc portfolio value at the beginning of the step & at the end to get reward
        old_cost = self.cost # cumulative trading costs
        old_option_value = self.generator.get_option_value(self.K, self.state[-1])
        old_und_value = self.state[1]
        
        # state: s -> s+1
        self.cost += self._get_trading_cost(action)
        if (action != self.state[0]) : self.trades += 1

        new_und = self.generator.get_next()
        self.state = [action, new_und, self.state[-1] - 1]
        new_option_value = self.generator.get_option_value(self.K, self.state[-1])
        
        # Calculate reward
        reward = self._get_reward(new_option_value, old_option_value, 
                                        new_und = self.state[1], 
                                        old_und = old_und_value, 
                                        trading_cost = self.cost - old_cost)

        if self.terminal:
            dic = self._out()
            df = pd.DataFrame(dic)
            return self.state, reward, self.terminal, {"output":df}

        # Add everything to memory if we're in testing phase
        if self.testing:
            self.option_memory.append(new_option_value)
            self.underlying_memory.append(self.state[1])
            self.cost_memory.append(self.cost)
            self.trades_memory.append(self.trades)
            self.holdings_memory.append(self.state[0])
            self.rewards_memory.append(reward)
            self.actions_memory.append(action)
            self.pnl_memory.append(- 100 * (new_option_value - old_option_value) + (self.state[1] - old_und_value) * self.state[0] - (self.cost - old_cost))
        
        return self.state, reward, self.terminal, {"output":pd.DataFrame()}

    def _get_trading_cost(self, action):
        return 0.5 * (abs(action-self.state[0]) + 0.01 * ((action - self.state[0])**2))

    def _get_reward(self, new_opt_val, old_opt_val, new_und, old_und, trading_cost):
        pnl = - 100 * (new_opt_val - old_opt_val) + (new_und - old_und) * self.state[0]
        return ( pnl - 0.5 * self.kappa * (pnl**2) ) - trading_cost
    
    def _out(self):
        if self.testing:
            dic = { "option" : self.option_memory,
                    "underlying" : self.underlying_memory,
                    "cost" : self.cost_memory,
                    "trades" : self.trades_memory,
                    "holdings": self.holdings_memory,
                    "rewards": self.rewards_memory,
                    "actions": self.actions_memory,
                    "pnl": self.pnl_memory}
        else: dic = {}
        return dic
    
    def reset(self):  
        """ Called at the beginning of each episode. """
        
        self.generator.reset()
        self.K = self.generator.current
        option_value = self.generator.get_option_value(self.K, self.expiry, True)
        
        # Reset some attributes
        self.cost = 0
        self.trades = 0
        self.terminal = False
        self.option_memory = [option_value]
        self.underlying_memory = [self.generator.current]
        self.cost_memory = [0.0]
        self.trades_memory = [0]
        self.rewards_memory = [0.0]
        self.holdings_memory = [0]
        self.actions_memory = [0.0]
        self.pnl_memory = [0.0]
        
        # Initiate state
        self.state = self._initiate_state()
        self.episode += 1
        return self.state

    def reset_with_seed(self, seed):
        state = self.reset()
        self.generator.reset_with_seed(seed)
        return state