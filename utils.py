import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from IPython.utils import io
from Environments import DiscreteEnv

class EpsFunction():
    def __init__(self, total_steps):
        self.total_steps = total_steps

    def get_func(self):

        def eps_func(step):
            return 1 - step / self.total_steps

        return eps_func

def plot_decisions(delta, df):
    # underlying & option values
    plt.figure(1, figsize=(12, 6))
    plt.subplot(121)
    plt.plot(delta.underlying)
    plt.title("underlying")
    plt.subplot(122)
    plt.plot(delta.option)
    plt.title("option value")

    delta_holdings = delta.holdings.values 
    delta_actions = delta.actions.values 
    model_holdings = df.holdings.values
    model_actions = df.actions.values

    plt.figure(2, figsize=(12, 8))
    # Holdings
    plt.subplot(211)
    plt.plot(delta_holdings, label='delta')
    plt.plot(model_holdings, label='model')
    plt.title("Holdings")
    plt.legend()

    # Actions
    plt.subplot(212)
    plt.plot(delta_actions, label='delta')
    plt.plot(model_actions, label='model')
    plt.title("Actions")
    plt.legend()

    plt.show()

def plot_pnl(delta, df):
    plt.figure(3, figsize=(12, 8))

    plt.subplot(221)
    plt.plot(delta.rewards, label='delta')
    plt.plot(df.rewards, label='model')
    plt.title("Rewards")
    plt.legend()


    plt.subplot(222)
    plt.plot(delta.pnl.cumsum(), label='delta')
    plt.plot(df.pnl.cumsum().values, label='model')
    plt.title("Cumulative PnL")
    plt.legend()

    plt.subplot(223)
    plt.plot(delta.cost, label='delta')
    plt.plot(df.cost, label='model')
    plt.title("Trading Costs")
    plt.legend()

    plt.subplot(224)
    plt.plot(delta.trades, label='delta')
    plt.plot(df.trades, label='model')
    plt.title("Num. of Trades")
    plt.legend()

    plt.show()

def plot_pnl_hist(pnl_paths_dict, pnl_dict, tcosts_dict, ntrades_dict):
    # Joint final PnL histograms
    plt.figure(1, figsize=(9, 6))
    binwidth = 20
    minimum = min(min(pnl_dict["model"]), min(pnl_dict["delta"]))
    maximum = max(max(pnl_dict["model"]), max(pnl_dict["delta"]))
    plt.hist(pnl_dict["model"], label="model", bins = np.arange(minimum, maximum + binwidth, binwidth))
    plt.hist(pnl_dict["delta"], label="delta", bins = np.arange(minimum, maximum + binwidth, binwidth))
    plt.title("PnL Histograms")
    plt.legend()

    # PnL paths graphs
    plt.figure(2, figsize=(18, 4))

    plt.subplot(121)
    for i in range(len(pnl_dict["model"])):
        plt.plot(pnl_paths_dict["model"][i].cumsum().values, color='r', alpha=0.4)
    plt.title("PnL paths Model")

    plt.subplot(122)
    for i in range(len(pnl_dict["delta"])):
        plt.plot(pnl_paths_dict["delta"][i].cumsum().values, color='r', alpha=0.4)
    plt.title("PnL paths Delta Hedge")


    # Trades and Costs histograms
    plt.figure(3, figsize=(12, 6))

    plt.subplot(121)
    binwidth = 20
    minimum = min(min(tcosts_dict["model"]), min(tcosts_dict["delta"]))
    maximum = max(max(tcosts_dict["model"]), max(tcosts_dict["delta"]))
    plt.hist(tcosts_dict["model"], label = "Model", bins = np.arange(minimum, maximum + binwidth, binwidth))
    plt.hist(tcosts_dict["delta"], label = "Delta Hedge", bins = np.arange(minimum, maximum + binwidth, binwidth))
    plt.title("Trading costs histogram")
    plt.legend()

    plt.subplot(122)
    binwidth = 1
    minimum = min(min(ntrades_dict["model"]), min(ntrades_dict["delta"]))
    maximum = max(max(ntrades_dict["model"]), max(ntrades_dict["delta"]))
    plt.hist(ntrades_dict["model"], label = "Model", bins = np.arange(minimum, maximum + binwidth, binwidth))
    plt.hist(ntrades_dict["delta"], label = "Delta Hedge", bins = np.arange(minimum, maximum + binwidth, binwidth))
    plt.title("Number of trades histogram")
    plt.legend()

    model_pnl_std = []
    delta_pnl_std = []
    for i in range(len(pnl_dict["model"])):
        model_pnl_std.append(np.std(pnl_paths_dict["model"][i]))
        delta_pnl_std.append(np.std(pnl_paths_dict["delta"][i]))


    plt.figure(4, figsize=(9,6))
    binwidth = 1
    minimum = min(min(model_pnl_std), min(delta_pnl_std))
    maximum = max(max(model_pnl_std), max(delta_pnl_std))
    plt.hist(model_pnl_std, label="model", bins = np.arange(minimum, maximum + binwidth, binwidth))
    plt.hist(delta_pnl_std, label="delta", bins = np.arange(minimum, maximum + binwidth, binwidth))
    plt.title("Histograms: Standard Deviation of PnL")
    plt.legend()

    plt.show()
    
    pd.DataFrame({"model": model_pnl_std, "delta":delta_pnl_std}).plot(kind='density')

def simulate_pnl(model_delta, n_steps, env_kwargs, simulator_func):
    pnl_paths_dict, pnl_dict, tcosts_dict, ntrades_dict = {"model" : [], "delta" : []}, {"model" : [], "delta" : []}, {"model" : [], "delta" : []}, {"model" : [], "delta" : []}
    env = DiscreteEnv(**env_kwargs)
    
    for i in tqdm(range(n_steps)):
        for key in ["model","delta"]:
            # Perform DRL testing
            env.reset_with_seed(11301*i)
            with io.capture_output() as _:
                if key == "model":
                    df = simulator_func(env)
                else:
                    df = model_delta.test(env)

            pnl_paths_dict[key].append(df.pnl)
            pnl_dict[key].append(df.pnl.cumsum().values[-1])
            tcosts_dict[key].append(df.cost.values[-1])
            ntrades_dict[key].append(df.trades.values[-1])

    return pnl_paths_dict, pnl_dict, tcosts_dict, ntrades_dict