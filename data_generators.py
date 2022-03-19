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

    def __init__(self, S0, r, sigma, freq, seed = None, barrier = None):
        self.initial = S0
        self.current = S0
        self.r = r # annualized drift
        self.sigma = sigma # annualized vol
        self.seed = seed
        self.T = 250 # number of trading days
        self.freq = freq # e.g. freq = 0.5 for trading twice a day
        self.dt = (1.0 / self.T) * freq # time increment
        self.H = barrier
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
        ttm = np.clip(ttm,0.0,np.inf)
        d1 = (np.log(self.current/K) + (self.r + self.sigma**2/2) * ttm ) / (self.sigma * sqrt(ttm))
        d2 = d1 - self.sigma * sqrt(ttm)
        if call:
            value = self.current * norm.cdf(d1) - K * exp( -self.r*ttm) * norm.cdf(d2)
        else:
            value = K * exp( -self.r*ttm) * norm.cdf(-d2) - self.current * norm.cdf(-d1)
        
        return value

    def get_barrier_value(self, K, ttm, up, out, is_knocked = False, call = True):
        """
        Calculates down & in barrier call/put price under current GBM dynamics.
        
        Parameters:
            - K = strike price
            - ttm  = time to maturity in periods
            - call = True/False, whether it's a call or a put
        """
        if is_knocked:
            if out:
                return 0.0
            else:
                return self.get_option_value(K, ttm, call)
        
        if ttm == 0:
            # not knocked call
            if call: return max(self.current - K, 0)*out
            # not knocked put
            else: return max(K - self.current, 0)*out

        orig_ttm = ttm
        ttm = ttm * self.dt # adjusting to annual terms
        ttm = np.clip(ttm,0.0,np.inf)

        lamb = 0.5 + self.r / (self.sigma ** 2)
        y = (np.log(self.H ** 2 / (self.current * K)) + lamb * ttm * (self.sigma ** 2) ) / (self.sigma * sqrt(ttm))
        
        x1 = (np.log(self.current / self.H) + lamb * ttm * (self.sigma ** 2) ) / (self.sigma * sqrt(ttm))
        y1 = (np.log(self.H / self.current) + lamb * ttm * (self.sigma ** 2) ) / (self.sigma * sqrt(ttm))
        
        # calls
        if call: 
            if up:
                if self.H > K:
                    up_in_call = self.current * norm.cdf(x1) \
                        - K * exp( -self.r*ttm) * norm.cdf(x1 - self.sigma * ttm) \
                        - self.current * ((self.H / self.current) ** (2*lamb)) * (norm.cdf(-y) - norm.cdf(-y1)) \
                        + K * exp( -self.r*ttm) * ((self.H / self.current) ** (2*lamb - 2)) * (norm.cdf(-y + self.sigma * ttm) - norm.cdf(-y1 + self.sigma * ttm))
                    
                    if out: return self.get_option_value(K, orig_ttm, call) - up_in_call
                    else: return up_in_call

                else:
                    if out: return 0.0
                    else: return self.get_option_value(K, orig_ttm, call)

            # down calls
            else:

                if self.H >= K:
                    down_out_call = self.current * norm.cdf(x1) \
                        - K * exp( -self.r*ttm) * norm.cdf(x1 - self.sigma * ttm) \
                        - self.current * ((self.H / self.current) ** (2*lamb)) * norm.cdf(y1) \
                        + K * exp( -self.r*ttm) * ((self.H / self.current) ** (2*lamb - 2)) * norm.cdf(y1 - self.sigma * ttm)
                    if out: return down_out_call
                    else: return self.get_option_value(K, orig_ttm, call) - down_out_call
    
                else:
                    down_in_call = self.current * ((self.H / self.current) ** (2*lamb)) * norm.cdf(y) - \
                        K * exp( -self.r*ttm) * ((self.H / self.current) ** (2*lamb - 2)) * \
                        norm.cdf(y - self.sigma * ttm)
                    if out: return self.get_option_value(K, orig_ttm, call) - down_in_call
                    else: return down_in_call

        # puts
        else:
            if up:
                if self.H > K:
                    up_in_put = - self.current * ((self.H / self.current) ** (2*lamb)) * norm.cdf(-y) \
                        + K * exp(-self.r*ttm) * ((self.H / self.current) ** (2*lamb - 2)) * norm.cdf(-y + self.sigma * ttm)
                    
                    if out: return self.get_option_value(K, orig_ttm, call) - up_in_put
                    else: return up_in_put

                else:
                    up_out_put = - self.current * norm.cdf(-x1) \
                        + K * exp(-self.r*ttm) * norm.cdf(-x1 + self.sigma * ttm) \
                        + self.current * ((self.H / self.current) ** (2*lamb)) * norm.cdf(-y1) \
                        - K * exp(-self.r*ttm) * ((self.H / self.current) ** (2*lamb - 2)) * norm.cdf(-y1 + self.sigma * ttm)
                    
                    if out: return up_out_put
                    else: return self.get_option_value(K, orig_ttm, call) - up_out_put

            # down puts
            else:

                if self.H >= K:
                    if out: return 0.0
                    else: return self.get_option_value(K, orig_ttm, call)
    
                else:
                    down_in_put = - self.current * norm.cdf(-x1) \
                        + K * exp( -self.r*ttm) * norm.cdf(-x1 + self.sigma * ttm) \
                        + self.current * ((self.H / self.current) ** (2*lamb)) * (norm.cdf(y) - norm.cdf(y1)) \
                        - K * exp( -self.r*ttm) * ((self.H / self.current) ** (2*lamb - 2)) * (norm.cdf(y - self.sigma * ttm) - norm.cdf(y1 - self.sigma * ttm))
                    
                    if out: return self.get_option_value(K, orig_ttm, call) - down_in_put
                    else: return down_in_put
        
    
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

class HestonGenerator:
    def __init__(self, S0, r, V0, rho, kappa, theta, xi, freq, seed = None):
        self.initial = S0
        self.current = S0
        self.r = r # annualized drift

        self.V0 = V0 # annualized starting variance
        self.rho = rho # correl BM
        self.kappa = kappa # mean reversion
        self.theta = theta # long-run vol mean
        self.xi = xi # vol of vol
        self.current_var = V0
        self.sigma = np.sqrt(V0)

        self.seed = seed
        self.T = 250 # number of trading days
        self.freq = freq # e.g. freq = 0.5 for trading twice a day
        self.dt = (1.0 / self.T) * freq # time increment

        if seed is not None:
            random.seed(seed)

    def get_next(self):
        multnorm = np.random.multivariate_normal([0,0],[[1,self.rho],[self.rho,1]])
        
        self.current_var = self.current_var + self.kappa*(self.theta - self.current_var)*self.dt + self.xi*np.sqrt(self.current_var*self.dt)*multnorm[1]
        self.current_var = np.abs(self.current_var)

        self.current = self.current * np.exp((self.r - 0.5*self.current_var)*self.dt + np.sqrt(self.current_var*self.dt)*multnorm[0])
        
        return self.current

    def get_option_value(self, K, ttm, call = True):
        """
        Calculates vanilla call/put price under GBM dynamics.
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

    def reset(self):
        self.current = self.initial
        self.current_var = self.V0
        if self.seed is not None:
            random.seed(self.seed)

    def reset_with_seed(self, seed):
        self.current = self.initial
        self.current_var = self.V0
        random.seed(seed)