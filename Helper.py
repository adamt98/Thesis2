from qfin import GeometricBrownianMotion, StochasticVarianceModel
import numpy as np

class MonteCarloCall:

    def simulate_price_gbm(self, strike, n, r, S, mu, sigma, dt, T):
        payouts = []
        for i in range(0, n):
            GBM = GeometricBrownianMotion(S, mu, sigma, dt, T)
            if(GBM.simulated_path[-1] >= strike):
                payouts.append((GBM.simulated_path[-1]-strike)*np.exp(-r*T))
            else:
                payouts.append(0)
        return np.average(payouts)

    def simulate_price_svm(self, strike, n, S, mu, r, div, alpha, beta, rho, vol_var, inst_var, dt, T):
        payouts = []
        for i in range(0, n):
            SVM = StochasticVarianceModel(S, mu, r, div, alpha, beta, rho, vol_var, inst_var, dt, T)
            if(SVM.simulated_paths[-1] >= strike):
                payouts.append((SVM.simulated_paths[-1]-strike)*np.exp(-r*T))
            else:
                payouts.append(0)
        return np.average(payouts)

    def __init__(self, strike, n, r, S, mu, sigma, dt, T, alpha=None, beta=None, rho=None, div=None, vol_var=None):
        if alpha is None:
            self.price = self.simulate_price_gbm(strike, n, r, S, mu, sigma, dt, T)
        else:
            inst_var = np.sqrt(sigma)
            self.price = self.simulate_price_svm(strike, n, S, mu, r, div, alpha, beta, rho, vol_var, inst_var, dt, T)

class MonteCarloPut:

    def simulate_price_gbm(self, strike, n, r, S, mu, sigma, dt, T):
        payouts = []
        for i in range(0, n):
            GBM = GeometricBrownianMotion(S, mu, sigma, dt, T)
            if(GBM.simulated_path[-1] <= strike):
                payouts.append((strike-GBM.simulated_path[-1])*np.exp(-r*T))
            else:
                payouts.append(0)
        return np.average(payouts)

    def simulate_price_svm(self, strike, n, S, mu, r, div, alpha, beta, rho, vol_var, inst_var, dt, T):
        payouts = []
        for i in range(0, n):
            SVM = StochasticVarianceModel(S, mu, r, div, alpha, beta, rho, vol_var, inst_var, dt, T)
            if(SVM.simulated_paths[-1] <= strike):
                payouts.append((strike-SVM.simulated_paths[-1])*np.exp(-r*T))
            else:
                payouts.append(0)
        return np.average(payouts)

    def __init__(self, strike, n, r, S, mu, sigma, dt, T, alpha=None, beta=None, rho=None, div=None, vol_var=None):
        if alpha is None:
            self.price = self.simulate_price_gbm(strike, n, r, S, mu, sigma, dt, T)
        else:
            inst_var = np.sqrt(sigma)
            self.price = self.simulate_price_svm(strike, n, S, mu, r, div, alpha, beta, rho, vol_var, inst_var, dt, T)

class MonteCarloBarrierPut:

    def simulate_price_gbm(self, strike, n, barrier, up, out, r, S, mu, sigma, dt, T):
        payouts = []
        for i in range(0, n):
            payouts = []
            for i in range(0, n):
                barrier_triggered = False
                GBM = GeometricBrownianMotion(S, mu, sigma, dt, T)
                for price in GBM.simulated_path:
                    if up:
                        if(price >= barrier):
                            barrier_triggered = True
                            break
                    elif not up:
                        if(price <= barrier):
                            barrier_triggered = True
                            break
                if out and not barrier_triggered:
                    if GBM.simulated_path[-1] <= strike:
                        payouts.append((strike - GBM.simulated_path[-1])*np.exp(-r*T))
                    else:
                        payouts.append(0)
                elif not out and barrier_triggered:
                    if GBM.simulated_path[-1] <= strike:
                        payouts.append((strike - GBM.simulated_path[-1])*np.exp(-r*T))
                    else:
                        payouts.append(0)
                else:
                    payouts.append(0)
            return np.average(payouts)

    def simulate_price_svm(self, strike, n, barrier, up, out, S, mu, r, div, alpha, beta, rho, vol_var, inst_var, dt, T):
        payouts = []
        for i in range(0, n):
            barrier_triggered = False
            SVM = StochasticVarianceModel(S, mu, r, div, alpha, beta, rho, vol_var, inst_var, dt, T)
            for price in SVM.simulated_paths:
                if up:
                    if(price >= barrier):
                        barrier_triggered = True
                        break
                elif not up:
                    if(price <= barrier):
                        barrier_triggered = True
                        break
            if out and not barrier_triggered:
                if SVM.simulated_paths[-1] <= strike:
                    payouts.append((strike - SVM.simulated_paths[-1])*np.exp(-r*T))
                else:
                    payouts.append(0)
            elif not out and barrier_triggered:
                if SVM.simulated_paths[-1] <= strike:
                    payouts.append((strike - SVM.simulated_paths[-1])*np.exp(-r*T))
                else:
                    payouts.append(0)
            else:
                payouts.append(0)
        return np.average(payouts)

    def __init__(self, strike, n, barrier, r, S, mu, sigma, dt, T, up=True, out=True, alpha=None, beta=None, rho=None, div=None, vol_var=None):
        if alpha is None:
            self.price = self.simulate_price_gbm(strike, n, barrier, up, out, r, S, mu, sigma, dt, T)
        else:
            inst_var = np.sqrt(sigma)
            self.price = self.simulate_price_svm(strike, n, barrier, up, out, S, mu, r, div, alpha, beta, rho, vol_var, inst_var, dt, T)