from Environments import DiscreteEnv
from Generators import GBM_Generator, HestonGenerator
from Models import DeltaHedge
import Utils

from typing import Callable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gym
import torch

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Figure
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

import QuantLib as ql

if __name__ == "__main__":
    # option inputs
    o = 100000
    maturity_date = ql.Date(o+30)
    spot_price = 100
    strike_price = 100
    barrier = 90
    volatility = 0.01*np.sqrt(250)
    dividend_rate = 0.0
    option_type = ql.Option.Put
    risk_free_rate = 0.0
    day_count = ql.Actual365Fixed()
    calculation_date = ql.Date(o)
    ql.Settings.instance().evaluationDate = calculation_date

    precalculation_date = ql.Date(o+1)

    # construct the option payoff
    european_option = ql.BarrierOption(ql.Barrier.DownIn, barrier, 0.0,
                        ql.PlainVanillaPayoff(option_type, strike_price),
                        ql.EuropeanExercise(maturity_date))

    # set the Heston parameters
    v0 = volatility*volatility # spot variance
    kappa = 1.5
    theta = v0
    hsigma = 0.1
    rho = -0.6
    spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot_price))

    #construct the Heston process
    # flat_ts = ql.YieldTermStructureHandle(ql.FlatForward(precalculation_date,
    #                                                     risk_free_rate, day_count))

    # dividend_yield = ql.YieldTermStructureHandle(ql.FlatForward(precalculation_date,
    #                                                             dividend_rate, day_count))
    # heston_process = ql.HestonProcess(flat_ts, dividend_yield,
    #                                 spot_handle, v0, kappa,
    #                                 theta, hsigma, rho)

    # # run the pricing engine
    # engine = ql.FdHestonBarrierEngine(ql.HestonModel(heston_process))
    # european_option.setPricingEngine(engine)

    # h_price = european_option.NPV()
    
    # print(h_price)

    gen = HestonGenerator(spot_price, strike_price, risk_free_rate, v0, rho, kappa, theta, xi=hsigma, barrier=barrier, expiry=30)
    
    b = []
    b1 = []
    b12 = []
    u = []
    count=0
    lengths = []
    for j in range(1000):
        length = 1
        gen.reset()
        for i in range(30):
            u1 = gen.get_next()
            if u1 < 97:
                count += 1
                lengths.append(length)
                break
            else: length +=1
            #bar = gen.get_barrier_value(strike_price, 1, False, False, False)
            #bar12 = gen.get_barrier_value(strike_price, 30, False, False, False)
            #bar1 = gen.get_GBM_barrier_value(strike_price, 1, False, False, False)
            #und = gen.get_next()
            #b.append(bar)
            #b1.append(bar1)
            #b12.append(bar12)
            #u.append(und)

    # plt.plot(b, 'b-')
    # plt.plot(b1, 'r-')
    # plt.plot(b12, 'g-')
    # plt.show()
    print("knocked in ", count/10, "%")
    print("avg. length ", np.mean(lengths))
