from Environments import DiscreteEnv
from Environments import BarrierEnv, BarrierEnv2, BarrierEnv3
from Generators import GBM_Generator
import Utils
import numpy as np
import pandas as pd
from Models import DeltaHedge
from stable_baselines3 import PPO, DQN

##### Environment config ###############

sigma = 0.01*np.sqrt(250) # 1% vol per day, annualized
r = 0.0 # Annualized
S0 = 100
freq = 0.2 #0.2 corresponds to trading freq of 5x per day
ttm = 50 # 50 & freq=0.2 => 10 days expiry
kappa = 0.1
cost_multiplier = 0.3
discount = 0.9

barrier = 97
n_puts_sold = 1
put_strike = 100
min_action = -100
max_action = 300
action_num = max_action - min_action
max_sellable_puts = 1

generator = GBM_Generator(S0, r, sigma, freq, barrier=barrier) # 
env_args = {
    "generator" : generator,
    "ttm" : ttm,
    "kappa" : kappa,
    "cost_multiplier" : cost_multiplier,
    "reward_type" : "static", # static
    "testing" : False,
    "n_puts_sold" : n_puts_sold,
    "min_action" : min_action,
    "max_action" : max_action,
    #"put_K" : put_strike,
    #"max_sellable_puts" : max_sellable_puts
}

def simulatePPO(env, obs):
    done = False
    while not done:
        action, _states = modelPPO.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
    
    return info['output']

def simulatePPO2(env, obs):
    done = False
    while not done:
        action, _states = modelPPO2.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
    
    return info['output']

def simulatePPO3(env, obs):
    done = False
    while not done:
        action, _states = modelPPO3.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
    
    return info['output']

# def simulateDQN(env, obs):
#     done = False
#     while not done:
#         action, _states = modelDQN.predict(obs, deterministic=True)
#         obs, reward, done, info = env.step(action)
    
#     return info['output']

if __name__ == "__main__":
    

    generator = GBM_Generator(S0, r, sigma, freq, seed=123, barrier=barrier) # 
    test_env_args = {
        "generator" : generator,
        "ttm" : ttm,
        "kappa" : kappa,
        "cost_multiplier" : cost_multiplier,
        "reward_type" : "static", # "static"
        "testing" : True,
        "n_puts_sold" : n_puts_sold,
        "min_action" : min_action,
        "max_action" : max_action,
        #"put_K" : put_strike,
        #"max_sellable_puts" : max_sellable_puts
    }

    test_envPPO = BarrierEnv2(**test_env_args)
    model = PPO(policy="MlpPolicy", env=test_envPPO)
    modelPPO = model.load('./networks/PPO/r2/c3/weights_PPO.zip')


    generator2 = GBM_Generator(S0, r, sigma, freq, seed=123, barrier=barrier) # 
    test_env_args2 = {
        "generator" : generator2,
        "ttm" : ttm,
        "kappa" : kappa,
        "cost_multiplier" : cost_multiplier,
        "reward_type" : "static", # "static" or "basic"
        "testing" : True,
        "n_puts_sold" : n_puts_sold,
        "min_action" : min_action,
        "max_action" : max_action,
        #"put_K" : put_strike,
        #"max_sellable_puts" : max_sellable_puts
    }

    test_envPPO2 = BarrierEnv2(**test_env_args2)
    model2 = PPO(policy="MlpPolicy", env=test_envPPO2)
    modelPPO2 = model2.load('./networks/PPO/r2/ttm20/weights_PPO.zip')


    generator3 = GBM_Generator(S0, r, sigma, freq, seed=123, barrier=barrier) # 
    test_env_args3 = {
        "generator" : generator3,
        "ttm" : 150,
        "kappa" : kappa,
        "cost_multiplier" : cost_multiplier,
        "reward_type" : "static", # "static"
        "testing" : True,
        "n_puts_sold" : n_puts_sold,
        "min_action" : min_action,
        "max_action" : max_action,
        #"put_K" : put_strike,
        #"max_sellable_puts" : max_sellable_puts
    }

    test_envPPO3 = BarrierEnv2(**test_env_args3)
    model3 = PPO(policy="MlpPolicy", env=test_envPPO3)
    modelPPO3 = model3.load('./networks/PPO/r2/ttm30/weights_PPO.zip')
    
    # test_envDQN = DiscreteEnv(**test_env_args)
    # model = DQN(policy="MlpPolicy", env=test_envDQN)
    # modelDQN = model.load('./networks/DQN/vanilla/c5/weights_DQN.zip')

    #####################################
    ####### TESTING PHASE ###############
    #####################################
    
    obs = test_envPPO.reset()
    dfPPO = simulatePPO(test_envPPO, obs)
    dfPPO["Agent"] = "PPO-TTM10"

    obs = test_envPPO2.reset()
    dfPPO2 = simulatePPO2(test_envPPO2, obs)
    dfPPO2["Agent"] = "PPO-TTM20"

    obs = test_envPPO3.reset()
    dfPPO3 = simulatePPO3(test_envPPO3, obs)
    dfPPO3["Agent"] = "PPO-TTM30"

    # obs = test_envDQN.reset()
    # dfDQN = simulateDQN(test_envDQN, obs)
    # dfDQN["Agent"] = "DQN"

    # delta hedge benchmark
    # delta_agent = DeltaHedge(generator.initial, call = False, n_puts_sold=n_puts_sold, min_action=min_action, put_K=put_strike) # , n_puts_sold=n_puts_sold, min_action=min_action &&& call->False # for barr3  put_K = put_strike
    # obs = test_envPPO.reset()
    # dfDelta = delta_agent.test(test_envPPO, obs)
    # dfDelta["Agent"] = "Delta"

    df = pd.concat([dfPPO, dfPPO2, dfPPO3], ignore_index=False) # dfDQN, 
    df['time'] = df.index
    df.reset_index(inplace=True)
    Utils.plot_decisions2(df)

    n_sim=100
    outPPO = Utils.simulate_pnl("PPO-TTM10", n_sim, test_envPPO, simulatePPO)
    outPPO2 = Utils.simulate_pnl("PPO-TTM20", n_sim, test_envPPO2, simulatePPO2)
    outPPO3 = Utils.simulate_pnl("PPO-TTM30", n_sim, test_envPPO3, simulatePPO3)
    #outDQN = Utils.simulate_pnl("DQN", n_sim, test_envDQN, simulateDQN)
    #outDelta = Utils.simulate_pnl("Delta", n_sim, test_envPPO, delta_agent.test)
    out = pd.concat([outPPO, outPPO2, outPPO3], ignore_index=True) # , outDQN
    Utils.plot_pnl_hist(out)
    Utils.perf_measures(out)
    # js = Utils.getJSDivergence(outPPO, outDelta)
    # print("JS divergence table PPO vs Delta:")
    # print(js.head())
    # js = Utils.getJSDivergence(outDQN, outDelta)
    # print("JS divergence table DQN vs Delta:")
    # print(js.head())
    # js = Utils.getJSDivergence(outPPO, outDQN)
    # print("JS divergence table PPO vs DQN:")
    # print(js.head())