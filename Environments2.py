import math
from Generators import GBM_Generator
from Reward import Reward
import numpy as np
import pandas as pd
import gym
from gym import spaces
from stable_baselines3.common.vec_env import DummyVecEnv

# no delta, fixed num. of puts sold
class BarrierEnv(gym.Env):

    def __init__(self,
                generator : GBM_Generator,
                ttm,
                kappa = 0.1,
                cost_multiplier = 0.5,
                testing = False,
                reward_type = "static",
                n_puts_sold = 1,
                min_action = -100,
                max_action = 100):
        
        # Init some constants
        self.expiry = ttm
        self.kappa = kappa
        self.terminal = False     
        self.testing = testing
        self.cost = 0
        self.trades = 0
        self.episode = 0
        self.K = generator.current # ATM
        self.cost_multiplier = cost_multiplier
        self._reward_class = Reward(kappa)
        self.reward_func = self._reward_class.get_reward_func(reward_type)

        self.generator = generator

        # Extra stuff needed for static-dynamic hedge
        self.n_puts_sold = n_puts_sold
        self.put_strike = self.K # same as DIP strike
        self.min_action = min_action
        self.max_action = max_action
        
        # State (observation) space for Gym [action, price, ttm]
        self.observation_space = spaces.Box(low = np.array([0, -20, 0]), high = np.array([1, 20, 1]) )

        # Create discrete action space
        #self.actions = np.arange(0, 101)
        self.action_space = spaces.Discrete(self.max_action-self.min_action+1)

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

        # Define payoff (to be changed)
        self.derivative_type = "barrier"
        self.call = False
        self.up = False
        self.out = False

        self.seed_n = 0

    def _get_derivative_value(self):
        if self.derivative_type == "option":
            return self.generator.get_option_value(self.K, self.state[-1], self.call)
        elif self.derivative_type == "barrier":
            return self.generator.get_barrier_value(self.K, self.state[-1], self.up, self.out, self.call)

    def _initiate_state(self):
        return [0, self.generator.current, self.expiry]

    def normalize_state(self, state):
        normalized = [state[0]/(self.max_action-self.min_action) , 10*(math.log(state[1]/100)-self.generator.r), state[2]/self.expiry]
        return normalized

    def denormalize_state(self, state):
        denorm = [state[0]*(self.max_action-self.min_action), 100*math.exp(state[1]/10.0+self.generator.r), state[2]*self.expiry]
        return denorm

    def step(self, action):
        # correct action
        action = action + self.min_action

        # calc portfolio value at the beginning of the step & at the end to get reward
        old_cost = self.cost # cumulative trading costs
        old_barrier_value = self._get_derivative_value()
        old_und_value = self.state[1]
        old_static_hedge_value = self.n_puts_sold * self.generator.get_option_value(self.put_strike, self.state[-1], call=False)
        
        # state: s -> s+1
        self.cost += self._get_trading_cost(action)
        if (action != self.state[0]) : self.trades += 1

        # Calc new portfolio value
        new_und = self.generator.get_next()
        self.state = [action, new_und, self.state[-1] - 1]
        new_barrier_value = self._get_derivative_value()
        new_static_hedge_value = self.n_puts_sold * self.generator.get_option_value(self.put_strike, self.state[-1], call=False)

        # Calculate reward
        reward, pnl = self.reward_func(new_barrier_value, old_barrier_value, new_static_hedge_value, old_static_hedge_value, 
                                        new_und = self.state[1], 
                                        old_und = old_und_value, 
                                        trading_cost = self.cost - old_cost,
                                        holdings = self.state[0])
        
        # check if terminal
        self.terminal = (self.state[-1] == 1) | self.generator.is_knocked

        if self.terminal:
            # at expiry need to liquidate our position in the underlying, in case knocked in -> liquidate and sell a single vanilla put
            liquidation_cost = self._get_trading_cost(0)
            reward -= liquidation_cost

            dic = self._out()
            df = pd.DataFrame(dic)
            # transform states
            state = self.normalize_state(self.state)
            return state, reward, self.terminal, {"output":df}

        # Add everything to memory if we're in testing phase
        if self.testing:
            self.option_memory.append(new_barrier_value)
            self.underlying_memory.append(self.state[1])
            self.cost_memory.append(self.cost)
            self.trades_memory.append(self.trades)
            self.holdings_memory.append(self.state[0])
            self.rewards_memory.append(reward)
            self.actions_memory.append(action)
            self.pnl_memory.append(pnl)
        
        # transform states
        state = self.normalize_state(self.state)
        return state, reward, self.terminal, {"output":pd.DataFrame()}

    def _get_trading_cost(self, action):
        return self.cost_multiplier * (abs(action-self.state[0]) + 0.01 * ((action - self.state[0])**2))
    
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
        #if self.generator.seed is None:
            #self.generator.reset_with_seed(self.seed_n)
            #self.seed_n = int(self.episode // 30)

        self.K = self.generator.current
        option_value = self._get_derivative_value()
        
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

        # transform states
        state = self.normalize_state(self.state)
        return state

    def reset_with_seed(self, seed):
        state = self.reset()
        self.generator.reset_with_seed(seed)
        return state

    def get_sb_env(self):
        ''' This function is called by the main file, initializes the whole environment. '''
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs

# with delta
class BarrierEnv2(gym.Env):

    def __init__(self,
                generator : GBM_Generator,
                ttm,
                kappa = 0.1,
                cost_multiplier = 0.5,
                testing = False,
                reward_type = "static",
                n_puts_sold = 1,
                min_action = -100,
                max_action = 100):
        
        # Init some constants
        self.expiry = ttm
        self.kappa = kappa
        self.terminal = False     
        self.testing = testing
        self.cost = 0
        self.trades = 0
        self.episode = 0
        self.K = generator.current # ATM
        self.cost_multiplier = cost_multiplier
        self._reward_class = Reward(kappa)
        self.reward_func = self._reward_class.get_reward_func(reward_type)

        self.generator = generator

        # Extra stuff needed for static-dynamic hedge
        self.n_puts_sold = n_puts_sold
        self.put_strike = self.K # same as DIP strike
        self.min_action = min_action
        self.max_action = max_action
        
        # State (observation) space for Gym [action, price, ttm, delta]
        self.observation_space = spaces.Box(low = np.array([0.0, -20.0, 0.0, -1.0]), high = np.array([1.0, 20.0, 1.0, 30.0]) )

        # Create discrete action space
        #self.actions = np.arange(0, 101)
        self.action_space = spaces.Discrete(self.max_action-self.min_action+1)

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

        # Define payoff (to be changed)
        self.derivative_type = "barrier"
        self.call = False
        self.up = False
        self.out = False

        self.seed_n = 0

    def _get_derivative_value(self):
        if self.derivative_type == "option":
            return self.generator.get_option_value(self.K, self.state[2], self.call)
        elif self.derivative_type == "barrier":
            return self.generator.get_barrier_value(self.K, self.state[2], self.up, self.out, self.call)

    def _get_current_delta(self, spot, ttm):
        return self.generator.get_DIP_delta(spot, self.K, ttm) - self.n_puts_sold*self.generator.get_delta(spot, self.put_strike, ttm)

    def _initiate_state(self):
        return [0, self.generator.current, self.expiry, self._get_current_delta(self.generator.current, self.expiry)]

    def normalize_state(self, state):
        normalized = [state[0]/(self.max_action-self.min_action) , 10*(math.log(state[1]/100)-self.generator.r), state[2]/self.expiry, state[3]]
        return normalized

    def denormalize_state(self, state):
        denorm = [state[0]*(self.max_action-self.min_action), 100*math.exp(state[1]/10.0+self.generator.r), state[2]*self.expiry, state[3]]
        return denorm

    def step(self, action):
        # correct action
        action = action + self.min_action

        # calc portfolio value at the beginning of the step & at the end to get reward
        old_cost = self.cost # cumulative trading costs
        old_barrier_value = self._get_derivative_value()
        old_und_value = self.state[1]
        old_static_hedge_value = self.n_puts_sold * self.generator.get_option_value(self.put_strike, self.state[2], call=False)
        
        # state: s -> s+1
        self.cost += self._get_trading_cost(action)
        if (action != self.state[0]) : self.trades += 1

        # Calc new portfolio value
        new_und = self.generator.get_next()
        self.state = [action, new_und, self.state[2] - 1, self._get_current_delta(self.state[1], self.state[2])]
        new_barrier_value = self._get_derivative_value()
        new_static_hedge_value = self.n_puts_sold * self.generator.get_option_value(self.put_strike, self.state[2], call=False)

        # Calculate reward
        reward, pnl = self.reward_func(new_barrier_value, old_barrier_value, new_static_hedge_value, old_static_hedge_value, 
                                        new_und = self.state[1], 
                                        old_und = old_und_value, 
                                        trading_cost = self.cost - old_cost,
                                        holdings = self.state[0])
        
        # check if terminal
        self.terminal = (self.state[2] == 1) | self.generator.is_knocked

        if self.terminal:
            # at expiry need to liquidate our position in the underlying, in case knocked in -> liquidate and sell a single vanilla put
            liquidation_cost = self._get_trading_cost(0)
            reward -= liquidation_cost

            dic = self._out()
            df = pd.DataFrame(dic)
            # transform states
            state = self.normalize_state(self.state)
            return state, reward, self.terminal, {"output":df}

        # Add everything to memory if we're in testing phase
        if self.testing:
            self.option_memory.append(new_barrier_value)
            self.underlying_memory.append(self.state[1])
            self.cost_memory.append(self.cost)
            self.trades_memory.append(self.trades)
            self.holdings_memory.append(self.state[0])
            self.rewards_memory.append(reward)
            self.actions_memory.append(action)
            self.pnl_memory.append(pnl)
        
        # transform states
        state = self.normalize_state(self.state)
        return state, reward, self.terminal, {"output":pd.DataFrame()}

    def _get_trading_cost(self, action):
        return self.cost_multiplier * (abs(action-self.state[0]) + 0.01 * ((action - self.state[0])**2))
    
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
        #if self.generator.seed is None:
            #self.generator.reset_with_seed(self.seed_n)
            #self.seed_n = int(self.episode // 30)

        self.K = self.generator.current
        option_value = self._get_derivative_value()
        
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

        # transform states
        state = self.normalize_state(self.state)
        return state

    def reset_with_seed(self, seed):
        state = self.reset()
        self.generator.reset_with_seed(seed)
        return state

    def get_sb_env(self):
        ''' This function is called by the main file, initializes the whole environment. '''
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs

# with delta & variable puts
class BarrierEnv3(gym.Env):

    def __init__(self,
                generator : GBM_Generator,
                ttm,
                kappa = 0.1,
                cost_multiplier = 0.5,
                testing = False,
                reward_type = "static",
                min_action = -100,
                max_action = 100,
                max_sellable_puts = 5):
        
        # Init some constants
        self.expiry = ttm
        self.kappa = kappa
        self.terminal = False     
        self.testing = testing
        self.cost = 0
        self.trades = 0
        self.episode = 0
        self.K = generator.current # ATM
        self.cost_multiplier = cost_multiplier
        self._reward_class = Reward(kappa)
        self.reward_func = self._reward_class.get_reward_func(reward_type)

        self.generator = generator

        # Extra stuff needed for static-dynamic hedge
        
        self.put_strike = self.K # same as DIP strike
        self.min_action = min_action
        self.max_action = max_action
        self.max_sellable_puts = max_sellable_puts
        
        # State (observation) space for Gym [action, price, ttm, delta, n_puts_sold]
        self.observation_space = spaces.Box(low = np.array([0.0, -20.0, 0.0, -1.0, 0.0]), high = np.array([1.0, 20.0, 1.0, 30.0, 1.0]) )

        # Create discrete action space: [underlying holdings, sold puts]
        self.action_space = spaces.MultiDiscrete([self.max_action-self.min_action+1, max_sellable_puts])
        
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

        # Define payoff (to be changed)
        self.derivative_type = "barrier"
        self.call = False
        self.up = False
        self.out = False

        self.seed_n = 0

    def _get_derivative_value(self):
        if self.derivative_type == "option":
            return self.generator.get_option_value(self.K, self.state[2], self.call)
        elif self.derivative_type == "barrier":
            return self.generator.get_barrier_value(self.K, self.state[2], self.up, self.out, self.call)

    def _get_current_delta(self, spot, ttm, n_puts_sold):
        return self.generator.get_DIP_delta(spot, self.K, ttm) - n_puts_sold*self.generator.get_delta(spot, self.put_strike, ttm)

    def _initiate_state(self):
        return [0, self.generator.current, self.expiry, self._get_current_delta(self.generator.current, self.expiry, 0)]

    def normalize_state(self, state):
        normalized = [state[0]/(self.max_action-self.min_action) , 10*(math.log(state[1]/100)-self.generator.r), state[2]/self.expiry, state[3], state[4]/self.max_sellable_puts]
        return normalized

    def denormalize_state(self, state):
        denorm = [state[0]*(self.max_action-self.min_action), 100*math.exp(state[1]/10.0+self.generator.r), state[2]*self.expiry, state[3], state[4]*self.max_sellable_puts]
        return denorm

    def step(self, action):
        # correct action
        action_und = action[0] + self.min_action

        # calc portfolio value at the beginning of the step & at the end to get reward
        old_cost = self.cost # cumulative trading costs
        old_barrier_value = self._get_derivative_value()
        old_und_value = self.state[1]
        old_static_hedge_value = self.state[4] * self.generator.get_option_value(self.put_strike, self.state[2], call=False)
        
        # state: s -> s+1
        self.cost += self._get_trading_cost(action_und) + self._get_trading_cost(action[1]) # 2nd one is for the puts
        
        if (action_und != self.state[0]) | (action[1] != self.state[4]) : self.trades += 1

        # Calc new portfolio value
        new_und = self.generator.get_next()
        self.state = [action, new_und, self.state[2] - 1, self._get_current_delta(self.state[1], self.state[2], action[1]), action[1]]
        new_barrier_value = self._get_derivative_value()
        new_static_hedge_value = self.state[4] * self.generator.get_option_value(self.put_strike, self.state[2], call=False)

        # Calculate reward
        reward, pnl = self.reward_func(new_barrier_value, old_barrier_value, new_static_hedge_value, old_static_hedge_value, 
                                        new_und = self.state[1], 
                                        old_und = old_und_value, 
                                        trading_cost = self.cost - old_cost,
                                        holdings = self.state[0])
        
        # check if terminal
        self.terminal = (self.state[2] == 1) | self.generator.is_knocked

        if self.terminal:
            # at expiry need to liquidate our position in the underlying, in case knocked in -> liquidate and sell a single vanilla put
            liquidation_cost = self._get_trading_cost(0)
            reward -= liquidation_cost

            dic = self._out()
            df = pd.DataFrame(dic)
            # transform states
            state = self.normalize_state(self.state)
            return state, reward, self.terminal, {"output":df}

        # Add everything to memory if we're in testing phase
        if self.testing:
            self.option_memory.append(new_barrier_value)
            self.underlying_memory.append(self.state[1])
            self.cost_memory.append(self.cost)
            self.trades_memory.append(self.trades)
            self.holdings_memory.append(self.state[0])
            self.rewards_memory.append(reward)
            self.actions_memory.append(action)
            self.pnl_memory.append(pnl)
        
        # transform states
        state = self.normalize_state(self.state)
        return state, reward, self.terminal, {"output":pd.DataFrame()}

    def _get_trading_cost(self, action):
        return self.cost_multiplier * (abs(action-self.state[0]) + 0.01 * ((action - self.state[0])**2))
    
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
        #if self.generator.seed is None:
            #self.generator.reset_with_seed(self.seed_n)
            #self.seed_n = int(self.episode // 30)

        self.K = self.generator.current
        option_value = self._get_derivative_value()
        
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

        # transform states
        state = self.normalize_state(self.state)
        return state

    def reset_with_seed(self, seed):
        state = self.reset()
        self.generator.reset_with_seed(seed)
        return state

    def get_sb_env(self):
        ''' This function is called by the main file, initializes the whole environment. '''
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs

