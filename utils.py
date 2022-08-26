import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyparsing import col
from tqdm import tqdm
from scipy.spatial.distance import jensenshannon as JS
from Environments import DiscreteEnv
import os
plots_dir = "./plots/"
import seaborn as sns

class EpsFunction():
    def __init__(self, total_steps):
        self.total_steps = total_steps

    def get_func(self):

        def eps_func(step):
            return 1 - step / self.total_steps

        return eps_func

def plot_decisions(delta, df):
    # initial stuff
    sns.set_theme(style="darkgrid")
    delta['time'] = delta.index
    df['time'] = df.index
    
    # Underlying value
    sns.relplot(data=delta, kind="line", x="time", y="underlying")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title("Underlying value")
    plt.savefig(plots_dir+"und_value.png", bbox_inches="tight")
    plt.show()

    # Option value
    sns.relplot(data=delta, kind="line", x="time", y="option")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title("Option value")
    plt.savefig(plots_dir+"opt_value.png", bbox_inches="tight")
    plt.show()

    # Holdings
    plt.plot(delta.holdings, label = "delta")
    plt.plot(df.holdings, label = "model")
    plt.xlabel("Time")
    plt.ylabel("Holdings")
    plt.title("Agent's position")
    plt.legend()
    plt.savefig(plots_dir+"holdings.png", bbox_inches="tight")
    plt.show()
    
    # Rewards
    plt.plot(delta.rewards, label='delta')
    plt.plot(df.rewards, label='model')
    plt.xlabel("Time")
    plt.ylabel("Reward")
    plt.title("Rewards")
    plt.legend()
    plt.savefig(plots_dir+"rewards.png", bbox_inches="tight")
    plt.show()

    # Cumulative PnL
    plt.plot(delta.pnl.cumsum(), label='delta')
    plt.plot(df.pnl.cumsum().values, label='model')
    plt.xlabel("Time")
    plt.ylabel("P&L")
    plt.title("Cumulative P&L")
    plt.legend()
    plt.savefig(plots_dir+"cumPnL.png", bbox_inches="tight")
    plt.show()

    # Costs
    plt.plot(delta.cost, label='delta')
    plt.plot(df.cost, label='model')
    plt.xlabel("Time")
    plt.ylabel("Cost")
    plt.title("Cumulative Trading Costs")
    plt.legend()
    plt.savefig(plots_dir+"cumCost.png", bbox_inches="tight")
    plt.show()

    # Number of Trades
    plt.plot(delta.trades, label='delta')
    plt.plot(df.trades, label='model')
    plt.xlabel("Time")
    plt.ylabel("Amount")
    plt.title("Cumulative Number of Trades")
    plt.legend()
    plt.savefig(plots_dir+"cumNTrades.png", bbox_inches="tight")
    plt.show()

def plot_decisions2(df : pd.DataFrame):
    # initial stuff
    sns.set_theme(style="darkgrid")
    
    
    # Underlying value
    dfDelta = df.query("Agent == 'Delta'")
    sns.relplot(data=dfDelta, kind="line", x="time", y="underlying")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title("Underlying value")
    plt.savefig(plots_dir+"und_value.png", bbox_inches="tight")
    plt.show()

    # Option value
    sns.relplot(data=dfDelta, kind="line", x="time", y="option")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title("Option value")
    plt.savefig(plots_dir+"opt_value.png", bbox_inches="tight")
    plt.show()

    # Holdings
    sns.relplot(data=df, kind="line", x="time", y="holdings", hue="Agent")
    plt.xlabel("Time")
    plt.ylabel("Holdings")
    plt.title("Agent's position")
    plt.legend()
    plt.savefig(plots_dir+"holdings.png", bbox_inches="tight")
    plt.show()
    
    # Rewards
    sns.relplot(data=df, kind="line", x="time", y="rewards", hue="Agent")
    plt.xlabel("Time")
    plt.ylabel("Reward")
    plt.title("Rewards")
    plt.legend()
    plt.savefig(plots_dir+"rewards.png", bbox_inches="tight")
    plt.show()

    # Cumulative PnL
    df["cumPnL"] = df.groupby(["Agent"]).pnl.cumsum()
    sns.relplot(data=df, kind="line", x="time", y="cumPnL", hue="Agent")
    plt.xlabel("Time")
    plt.ylabel("P&L")
    plt.title("Cumulative P&L")
    plt.legend()
    plt.savefig(plots_dir+"cumPnL.png", bbox_inches="tight")
    plt.show()

    # Costs
    sns.relplot(data=df, kind="line", x="time", y="cost", hue="Agent")
    plt.xlabel("Time")
    plt.ylabel("Cost")
    plt.title("Cumulative Trading Costs")
    plt.legend()
    plt.savefig(plots_dir+"cumCost.png", bbox_inches="tight")
    plt.show()

    # Number of Trades
    sns.relplot(data=df, kind="line", x="time", y="trades", hue="Agent")
    plt.xlabel("Time")
    plt.ylabel("Amount")
    plt.title("Cumulative Number of Trades")
    plt.legend()
    plt.savefig(plots_dir+"cumNTrades.png", bbox_inches="tight")
    plt.show()

def plot_decisions_extra(delta,df):
    # initial stuff
    delta['time'] = delta.index
    df['time'] = df.index

    # Holdings
    plt.plot(delta.actions_opt, label = "delta")
    plt.plot(df.actions_opt, label = "model")
    plt.xlabel("Time")
    plt.ylabel("Option's Holdings")
    plt.title("Agent's position")
    plt.legend()
    plt.savefig(plots_dir+"holdings_opt.png", bbox_inches="tight")
    plt.show()

def plot_pnl_hist(df):

    sns.displot(df, x="P&L", hue="Agent", kind="kde", fill=True, bw_adjust = 1.0)
    plt.title("Smoothed P&L Density")
    plt.savefig(plots_dir+"pnl_hist.png", bbox_inches="tight")
    plt.show()

    # PnL paths graphs
    # plt.figure(2, figsize=(18, 4))

    # plt.subplot(121)
    # for i in range(len(pnl_dict["model"])):
    #     plt.plot(pnl_paths_dict["model"][i].cumsum().values, color='r', alpha=0.4)
    # plt.title("PnL paths Model")

    # plt.subplot(122)
    # for i in range(len(pnl_dict["delta"])):
    #     plt.plot(pnl_paths_dict["delta"][i].cumsum().values, color='r', alpha=0.4)
    # plt.title("PnL paths Delta Hedge")
    # plt.savefig(plots_dir+"pnl_paths.png")

    # Trading costs histogram
    sns.displot(df, x="Trading Cost", hue="Agent", kind="kde", fill=True, bw_adjust = 1.0, cut=0)
    plt.title("Trading Costs Density")
    plt.savefig(plots_dir+"cost_hist.png", bbox_inches="tight")
    plt.show()
    
    # Trades histogram
    sns.displot(df, x="Number of Trades", hue="Agent", stat="density", fill=True)
    plt.title("Number of trades histogram")
    plt.savefig(plots_dir+"ntrades_hist.png", bbox_inches="tight")
    plt.show()
    
    # P&L Volatility
    sns.displot(df, x="P&L Volatility", hue="Agent", kind="kde", fill=True, bw_adjust = 1.0, cut=0)
    plt.title("P&L Volatility")
    plt.savefig(plots_dir+"std_hist.png", bbox_inches="tight")
    plt.show()

def simulate_pnl(name, n_steps, env, simulator_func):
    data = []
    for i in tqdm(range(n_steps)):
        obs = env.reset_with_seed(11301*i)
        df = simulator_func(env, obs)
        data.append([df.pnl.cumsum().values[-1], df.cost.values[-1], df.trades.values[-1], df.pnl, df.pnl.std(), name])

    out = pd.DataFrame(data, columns=["P&L","Trading Cost","Number of Trades", "P&L Paths", "P&L Volatility", "Agent"])
    return out

def perf_measures(df : pd.DataFrame):
    # dist mean, stdev, min, max, skew, kurtosis, VaR
    def getStats(df, colName):
        pnl_measures = {
            "mean" : df[colName].mean(),
            "volatility": df[colName].std(),
            "min": df[colName].min(),
            "max": df[colName].max(),
            "skew": df[colName].skew(), 
            "kurtosis": df[colName].kurtosis(),
            "VaR.05": df[colName].quantile(0.05)
        }
        return pd.DataFrame(pnl_measures, index=[0])

    # final PnL 
    pnlStats = df.groupby("Agent").apply(lambda x : getStats(x, "P&L"))

    # tcosts 
    tcostStats = df.groupby("Agent").apply(lambda x : getStats(x, "Trading Cost"))
    
    # nTrades 
    nTradesStats = df.groupby("Agent").apply(lambda x : getStats(x, "Number of Trades"))
    
    # Volatility 
    volStats = df.groupby("Agent").apply(lambda x : getStats(x, "P&L Volatility"))
    

    print("P&L:")
    print(pnlStats.head(10))
    print("T-cost stats:")
    print(tcostStats.head(10))
    print("n trades stats:")
    print(nTradesStats.head(10))
    print("Vol stats:")
    print(volStats.head(10))
    return None

def getJSDivergence(df1, df2):
    def getDiv(colName):
        binned1, _ = np.histogram(df1[colName], bins=20)
        binned2, _ = np.histogram(df2[colName], bins=20)
        return JS(binned1, binned2)**2

    cols = ["P&L","Trading Cost","Number of Trades","P&L Volatility"]
    vals = [getDiv(colName) for colName in cols]
    return pd.DataFrame(vals, cols)

