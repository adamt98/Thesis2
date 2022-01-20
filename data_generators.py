import numpy as np
import pandas as pd
import random
from random import gauss
from math import sqrt, exp
from scipy.stats import norm

class GBM_Generator:
    """Provides methods for getting GBM simulated stock price

    Attributes
    ----------
    S0:      Asset inital price.
    r:       Interest rate expressed in annual terms.
    sigma:   Volatility expressed annual terms. 
    

    Methods
    -------
    get_next()
        generates next value
        
    reset()
        resets the generator to intial value (S0)

    """

    def __init__(self, S0, r, sigma, freq, seed = None):
        self.initial = S0
        self.current = S0
        self.r = r # annualized drift
        self.sigma = sigma # annualized vol
        self.seed = seed
        self.T = 250 # number of trading days
        self.freq = freq # e.g. freq = 0.5 for trading twice a day
        self.dt = (1.0 / self.T) * freq # time increment

        if seed is not None:
            random.seed(seed)
        

    def get_next(self):
        self.current = self.current * exp((self.r - 0.5 * self.sigma ** 2) * self.dt + \
                                          self.sigma * sqrt(self.dt) * gauss(mu=0, sigma=1))
        
        return self.current
    
    def get_option_value(self, K, ttm, call = True):
        """
        Calculates vanilla call/put price under current GBM dynamics.
        
        Parameters:
            - K = strike price
            - ttm  = time to maturity in periods
            - call = True/False, whether it's a call or a put
        """
        if ttm == 0:
            if call: return max(self.current - K, 0)
            else: return max(K - self.current, 0)

        ttm = ttm * self.dt # adjusting to annual terms
        d1 = (np.log(self.current/K) + (self.r + self.sigma**2/2) * ttm ) / (self.sigma * sqrt(ttm))
        d2 = d1 - self.sigma * sqrt(ttm)
        if call:
            value = self.current * norm.cdf(d1) - K * exp( -self.r*ttm) * norm.cdf(d2)
        else:
            value = K * exp( -self.r*ttm) * norm.cdf(-d2) - self.current * norm.cdf(-d1)
        
        return value
    
    def get_delta(self, spot, K, ttm):
        ttm = ttm * self.dt # adjusting to annual terms
        d1 = (np.log(spot/K) + (self.r + self.sigma**2/2) * ttm ) / (self.sigma * sqrt(ttm))
        return norm.cdf(d1)
    
    def reset(self):
        self.current = self.initial
        if self.seed is not None:
            random.seed(self.seed)

    def reset_with_seed(self, seed):
        self.current = self.initial
        random.seed(seed)