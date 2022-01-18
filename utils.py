import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from discrete_environments import DiscreteEnv

from IPython.utils import io
import matplotlib.pyplot as plt

class EpsFunction():
    def __init__(self, total_steps):
        self.total_steps = total_steps

    def get_func(self):

        def eps_func(step):
            return 1 - step / self.total_steps

        return eps_func

def plot_decisions(delta, df):
    # underlying & option values
    plt.figure(1, figsize=(18, 6))
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

    plt.figure(2, figsize=(18, 12))
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


# report PnL paths, final PnL distribution
# final trading costs & #ofTrades distribution
# same for delta hedge
def simulate_pnl( model, model_delta, n_steps, env_kwargs):
    pnl_paths_dict, pnl_dict, tcosts_dict, ntrades_dict = {"model" : [], "delta" : []}, {"model" : [], "delta" : []}, {"model" : [], "delta" : []}, {"model" : [], "delta" : []}

    env = DiscreteEnv(**env_kwargs)
    delta_env = DiscreteEnv(**env_kwargs)
    
    sanity_check = {"model":0, "delta":0, "prev":-1}

    for i in tqdm(range(n_steps)):
        for key in ["model","delta"]:
            # Perform DRL testing
            env.reset_with_seed(11301*i)
            delta_env.reset_with_seed(11301*i) 
            with io.capture_output() as captured:
                if key == "model":
                    df = model.test(env)
                else:
                    df = model_delta.test(delta_env)

            pnl_paths_dict[key].append(df.pnl) # cumsum deleted
            pnl_dict[key].append(df.pnl.cumsum().values[-1])
            tcosts_dict[key].append(df.cost.values[-1])
            ntrades_dict[key].append(df.trades.values[-1])
            sanity_check[key] = df.underlying.values[-1]

        assert sanity_check["model"] == sanity_check["delta"], "Generated different paths for model and delta hedge"
        assert sanity_check["model"] != sanity_check["prev"], "Generated same paths for 2 consecutive runs"
        assert sanity_check["delta"] != sanity_check["prev"], "Generated same paths for 2 consecutive runs"

    return pnl_paths_dict, pnl_dict, tcosts_dict, ntrades_dict

def plot_pnl_hist(pnl_paths_dict, pnl_dict, tcosts_dict, ntrades_dict):
    # Joint final PnL histograms
    plt.figure(1, figsize=(18, 12))
    plt.hist(pnl_dict["model"], label="model")
    plt.hist(pnl_dict["delta"], label="delta")
    plt.title("PnL Histograms")
    plt.legend()

    # PnL paths graphs
    plt.figure(2, figsize=(18, 6))

    plt.subplot(121)
    for i in range(len(pnl_dict["model"])):
        plt.plot(pnl_paths_dict["model"][i].cumsum().values, color='r', alpha=0.4)
    plt.title("PnL paths Model")

    plt.subplot(122)
    for i in range(len(pnl_dict["delta"])):
        plt.plot(pnl_paths_dict["delta"][i].cumsum().values, color='r', alpha=0.4)
    plt.title("PnL paths Delta Hedge")


    # Trades and Costs histograms
    plt.figure(3, figsize=(18, 6))

    plt.subplot(121)
    plt.hist(tcosts_dict["model"], label = "Model")
    plt.hist(tcosts_dict["delta"], label = "Delta Hedge")
    plt.title("Trading costs histogram")
    plt.legend()

    plt.subplot(122)
    plt.hist(ntrades_dict["model"], label = "Model")
    plt.hist(ntrades_dict["delta"], label = "Delta Hedge")
    plt.title("Number of trades histogram")
    plt.legend()

    model_pnl_std = []
    delta_pnl_std = []
    for i in range(len(pnl_dict["model"])):
        model_pnl_std.append(np.std(pnl_paths_dict["model"][i]))
        delta_pnl_std.append(np.std(pnl_paths_dict["delta"][i]))


    plt.figure(4, figsize=(18, 12))
    plt.hist(model_pnl_std, label="model")
    plt.hist(delta_pnl_std, label="delta")
    plt.title("Histograms: Standard Deviation of PnL")
    plt.legend()


    plt.show()
    
    pd.DataFrame({"model": model_pnl_std, "delta":delta_pnl_std}).plot(kind='density')



def custom_schedule(initial_value, mid_value, final_value, initial_cutoff):
    scaling_const = 1 / (1 - initial_cutoff)
    def func(progress_remaining: float) -> float:
        if (1-progress_remaining) < initial_cutoff:
            return initial_value
        else:
            return scaling_const * progress_remaining * (mid_value - final_value) + final_value

    return func


def simulate_pnl2( model_delta, n_steps, env_kwargs):
    pnl_paths_dict, pnl_dict, tcosts_dict, ntrades_dict = {"model" : [], "delta" : []}, {"model" : [], "delta" : []}, {"model" : [], "delta" : []}, {"model" : [], "delta" : []}

    #env = DiscreteEnv(**env_kwargs)
    delta_env = DiscreteEnv(**env_kwargs)
    
    sanity_check = {"model":0, "delta":0, "prev":-1}

    for i in range(n_steps):
        for key in ["delta"]:
            # Perform DRL testing
            delta_env.reset_with_seed(11301*i)        # !!!!!!!!!! forgot to seed delta env
            with io.capture_output() as captured:
                # if key == "model":
                #     df = model.test(env)
                # else:
                df = model_delta.test(delta_env)

            pnl_paths_dict[key].append(list(df.pnl.values)) # !!!!!!!!! possibly change here too .cumsum().values[1:]
            pnl_dict[key].append(df.pnl.cumsum().values[-1])
            tcosts_dict[key].append(df.cost.values[-1])
            ntrades_dict[key].append(df.trades.values[-1])
            sanity_check[key] = df.underlying.values[-1]

        # assert sanity_check["model"] == sanity_check["delta"], "Generated different paths for model and delta hedge"
        # assert sanity_check["model"] != sanity_check["prev"], "Generated same paths for 2 consecutive runs"
        # assert sanity_check["delta"] != sanity_check["prev"], "Generated same paths for 2 consecutive runs"

    return pnl_paths_dict, pnl_dict, tcosts_dict, ntrades_dict


def plot_pnl_hist2(pnl_paths_dict, pnl_dict, tcosts_dict, ntrades_dict):
        # Trades and Costs histograms
    plt.figure(1, figsize=(18, 6))

    plt.subplot(121)
    #plt.hist(tcosts_dict["model"], label = "Model")
    plt.hist(tcosts_dict["delta"], label = "Delta Hedge")
    plt.title("Trading costs histogram")
    plt.legend()

    plt.subplot(122)
    #plt.hist(ntrades_dict["model"], label = "Model")
    plt.hist(ntrades_dict["delta"], label = "Delta Hedge")
    plt.title("Number of trades histogram")
    plt.legend()

    #model_pnl_std = []
    delta_pnl_std = []
    for i in range(len(pnl_dict["delta"])):
        #model_pnl_std.append(np.std(pnl_paths_dict["model"][i]))

        path = np.array(pnl_paths_dict["delta"][i])
        #rets = np.log( np.divide(path[1:], path[:-1]) - 1 )
        delta_pnl_std.append(np.std(path))

    


    plt.figure(2, figsize=(18, 12))
    #plt.hist(model_pnl_std, label="model")
    plt.hist(delta_pnl_std, label="delta", bins = 30)
    plt.title("Histograms: Standard Deviation of PnL")
    plt.legend()
    plt.show()

    
