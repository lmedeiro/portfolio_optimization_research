import sys
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cvxpy as cp

import sklearn.covariance as sk_cov
from pyRMT import pyRMT as rmt
from pdb import set_trace as bp

cw = os.getcwd()
# print(cw)
global OPTIMUM_CONTAINER
OPTIMUM_CONTAINER = {'date': [], 'value': []}
ACCEPTED_STRATEGIES = ['gmvp','mdp_original',
                       'mdp_kappa', 'mdp_D', 'one_over_n']

if 'portfolio_research_project' in cw:
    home_dir = os.path.expandvars("$HOME")
    app_src_dir = '/dev/repos/FinancialAnalyticsSystem/src/PortfolioManagementSystem/portfolio_research_project'
    sys.path.insert(0, home_dir + app_src_dir)
    from BackTestingSystem import bt
else:
   from src.PortfolioManagementSystem.portfolio_research_project.BackTestingSystem import bt


def calculate_Sigma(X, method_name='sample'):
    sigma = None
    if method_name == 'sample':
        sigma = np.cov(X.values, rowvar=False)
    elif method_name == 'ledoit-wolf':
        sigma = sk_cov.ledoit_wolf(X.values)[0]
    elif method_name == 'rie' or method_name == 'RIE':
        # bp()
        if X.values.shape[0] <= X.values.shape[1]:
            # sigma = sk_cov.ledoit_wolf(X.values)[0]
            sigma = np.cov(X.values, rowvar=False)
        else:
            sigma = rmt.optimalShrinkage(X.values, return_covariance=True)

    elif method_name == 'clipped':
        if X.values.shape[0] <= X.values.shape[1]:
            sigma = sk_cov.ledoit_wolf(X.values)[0]
            # sigma = rmt.optimalShrinkage(X.values, return_covariance=True)
        else:
            sigma = rmt.clipped(X.values, return_covariance=True)
    else:
        raise Exception('Error occurred with method name: {}'.format(method_name))

    return sigma


def calculate_gmvp(sigma):
    """
    General Mean Variance Portfolio
    :param sigma:
    :return:
    """
    N = sigma.shape[0]
    x = cp.Variable(shape=(N,1))
    problem = cp.Problem(cp.Minimize(cp.quad_form(x, sigma)),
                         [x >= np.ones([N,1]) * 0,
                          np.ones([N,1]).T @ x == 1])
    optimum = problem.solve()
    return x.value, optimum


def calculate_one_over_n(sigma):
    N = sigma.shape[0]
    volatilities = np.sqrt(np.diag(sigma))
    volatilities = volatilities.reshape(volatilities.shape[0], 1)
    w_N = np.ones([N, 1]) * 1 / N
    V_N = w_N.T @ volatilities
    D = V_N / np.sqrt(w_N.T @ sigma @ w_N)
    return w_N, D


def calculate_mdp_original(sigma):
    """
    Maximum Diversified Portfolio, proposed in Fall 2008, without parameters
    :param sigma: covariance matrix
    :return: weight matrix M, diversification ratio D
    """
    N = sigma.shape[0]
    # bp()
    volatilities = np.sqrt(np.diag(sigma))
    A = np.linalg.lstsq(sigma, volatilities, rcond=None)[0]
    B = np.ones([N, 1]).T @ A
    M = A / B
    M = M.reshape([N, 1])
    D = (M.T @ volatilities) / np.sqrt(M.T @ sigma @ M)
    return M, D


def calculate_mdp_based_on_kappa(sigma):
    """
    Maximum Diversified Portfolio, proposed in Fall 2008
    :param sigma: covariance matrix
    :param min_gmvp_point: paramter that stipulates what is the minimum gmpv point at which
    to place as constraint in the QCP
    :return:
    """
    N = sigma.shape[0]
    ones = np.ones([N, 1])
    inv_sigma = np.linalg.inv(sigma)

    volatilities = np.sqrt(np.diag(sigma)) * np.identity(N)
    inv_volatilities = np.linalg.inv(volatilities)
    kappa = cp.Variable(shape=[1, 1])
    M = kappa * inv_volatilities @ inv_sigma @ ones
    # D = cp.Variable(shape=(1, 1))

    constraints = [M >= np.zeros([N, 1]),
                   ones.T @ M == 1,
                   kappa >= 0,
                   ]
    problem = cp.Problem(cp.Minimize(kappa),
                         constraints)
    optimum = problem.solve(solver='ECOS',
                            qcp=True,
                            verbose=False
                            )
    M = M.value
    # try:
    D = M.T @ np.diag(volatilities) / np.sqrt(M.T @ sigma @ M)
    return M, D
    # except:
    #     # bp()
    #     # print('compute 1/n instead of mdp_original')
    #
    #     volatilities = np.sqrt(np.diag(sigma))
    #     volatilities = volatilities.reshape(volatilities.shape[0], 1)
    #     w_N = np.ones([N, 1]) * 1 / N
    #     V_N = w_N.T @ volatilities
    #     D = V_N / np.sqrt(w_N.T @ sigma @ w_N)
    #     return w_N, D


def calculate_mdp_based_on_D(sigma, D_N=3):
    """
    Maximum Diversified Portfolio based on a given D_N. If D_N is not given, it will be based
    on a factor of the diversified ratio of the 1/N portfolio strategy.
    :param sigma: covariance matrix
    :param min_gmvp_point: paramter that stipulates what is the minimum gmpv point at which
    to place as constraint in the QCP
    :return: x: weghts for assets, D: diversification ratio
    """
    N = sigma.shape[0]
    # bp()
    volatilities = np.sqrt(np.diag(sigma))
    volatilities = volatilities.reshape(volatilities.shape[0], 1)
    w_N = np.ones([N, 1]) * 1 / N
    V_N = w_N.T  @ volatilities
    if D_N is None:
        D_N = V_N / np.sqrt(w_N.T @ sigma @ w_N) * 1.5
        # D_N = 3
    R_N = V_N**2 / D_N**2
    x = cp.Variable(shape=(N,1))
    V = x.T @ volatilities # cp.quad_form(x, sigma)
    constraints = [x >= np.ones([N,1]) * 0,
                   np.ones([N,1]).T @ x == 1,
                   cp.quad_form(x, sigma) <= R_N,
                   ]
    problem = cp.Problem(cp.Maximize(V),
                         constraints)
    optimum = problem.solve(qcp=True)
    gmvp_point = cp.quad_form(x, sigma).value
    # if we couldn't find an optimum point, just allocated money equally.
    # TODO: Investigate what happens when the algorithm receives an array that is not correct.
    try:
        x = np.array(x.value, dtype=float)
        D = V.value / np.sqrt(x.T @ sigma @ x)

    except:
        # bp()
        # print('computer 1/n instead of mdp_new')
        x = w_N
        D = V_N / np.sqrt(x.T @ sigma @ x)
    return x, D


def calculate_um(sigma):
    #TODO: Not yet properly implemented
    N = sigma.shape[0]
    x = cp.Variable(shape=(N,1))
    problem = cp.Problem(cp.Minimize((1/2) * cp.quad_form(x, sigma)),
                         [x >= np.ones([N,1]) * 0,
                          np.ones([N,1]).T @ x == 1])
    optimum = problem.solve()
    return x.value, optimum


class WeighOptimization(bt.Algo):

    """
    Sets temp['weights'] based on an optimization chosen by the user. Default method is Mean-Variance.

    Sets the target weights based on ffn's calc_mean_var_weights. This is a
    Python implementation of Markowitz's mean-variance optimization.

    See:
        http://en.wikipedia.org/wiki/Modern_portfolio_theory#The_efficient_frontier_with_no_risk-free_asset

    Args:
        * lookback (DateOffset): lookback period for estimating volatility
        * bounds ((min, max)): tuple specifying the min and max weights for
            each asset in the optimization.
        * covar_method (str): method used to estimate the covariance. See ffn's
            calc_mean_var_weights for more details.
            In addition, RIE and clipping eigenvalues method are also provided for comparison.
        * rf (float): risk-free rate used in optimization.
        * lag (DateOffset): Number of days to use as the reference return. For example, if there is a
        lag of 3 days, that means that the returns are based on three days prior.

    Sets:
        * weights

    Requires:
        * selected

    """

    def __init__(self, lookback=pd.DateOffset(months=3),
                 bounds=(0., 1.), covar_method='ledoit-wolf',
                 rf=0., lag=pd.DateOffset(days=0), optimum_container=None,
                 optimization_method='gmvp', D_N=3):
        super(WeighOptimization, self).__init__()
        self.lookback = lookback
        self.lag = lag
        self.bounds = bounds
        self.covar_method = covar_method
        self.optimization_method = optimization_method
        self.optimum_container = optimum_container
        self.rf = rf
        self.D_N = D_N

    def __call__(self, target):
        selected = target.temp['selected']

        if len(selected) == 0:
            target.temp['weights'] = {}
            return True

        if len(selected) == 1:
            target.temp['weights'] = {selected[0]: 1.}
            return True

        t0 = target.now - self.lag
        prc = target.universe[selected].loc[t0 - self.lookback:t0]
        # It calculates the returns every time we compute the weights again.
        returns = prc.to_returns().dropna()
        if returns.index.size > 1:
            returns = returns.dropna()
            sigma = calculate_Sigma(returns, method_name=self.covar_method)
            if self.optimization_method == 'gmvp':
                raw_weights, optimum = calculate_gmvp(sigma)
            elif self.optimization_method == 'mdp_original':
                raw_weights, optimum = calculate_mdp_original(sigma)
            elif self.optimization_method == 'mdp_kappa':
                raw_weights, optimum = calculate_mdp_based_on_kappa(sigma)
            elif self.optimization_method == 'mdp_D':
                raw_weights, optimum = calculate_mdp_based_on_D(sigma, D_N=self.D_N)
            elif self.optimization_method == 'one_over_n':
                raw_weights, optimum = calculate_one_over_n(sigma)

            else:
                raise Exception('Optimization method not implemented')
                raw_weights = None
            #for index, weight in enumerate(raw_weights):
            #    if weight < 0.01:
            #        raw_weights[index] = 0.0
            tw = pd.DataFrame(data=raw_weights.T, columns=prc.columns)
            target.temp['weights'] = tw.dropna()
            # bp()
            if OPTIMUM_CONTAINER:
                OPTIMUM_CONTAINER['date'].append(t0)
                OPTIMUM_CONTAINER['value'].append(optimum)
        else:
            n = len(selected)

            if n == 0:
                target.temp['weights'] = {}
            else:
                w = 1.0 / n
                target.temp['weights'] = {x: w for x in selected}
        return True


def get_available_strategies(lookback, lag, covar_method, strategy_name,
                             optimum_container=None, D_N=3):
    """
    Helper function to build test. This function helps include test strategies for
    every parameter passed.
    :param lookback:
    :param lag:
    :param covar_method:
    :param strategy_name:
    :return:
    """
    # bp()
    if strategy_name in ACCEPTED_STRATEGIES:
        return WeighOptimization(lookback=lookback,
                                 lag=lag,
                                 covar_method=covar_method,
                                 optimization_method=strategy_name,
                                 optimum_container=optimum_container,
                                 D_N=D_N)
    elif strategy_name == 'random':
        return bt.algos.WeighRandomly()
    else:
        raise Exception("Did not understand strategy")


def build_test(number_of_assets, data_container,
               optimum_container=None, covariance_methods=['ledoit-wolf'],
               weight_strategy_names=['gmvp'], commission_functions=[None],
               lookback_periods=[pd.DateOffset(months=6)], lag_times=[pd.DateOffset(months=0)],
               add_random_strategy=False, add_one_over_n_strategy=False, D_N=3,
               ):
    """
    Function that helps build tests. Given a certain set of parameters, a strategy container and test
    container will be built and returned, so the user can run the tests.
    :param number_of_assets:
    :param data_container:
    :param covariance_methods:
    :param weight_strategy_names:
    :param commission_functions:
    :param lookback_periods:
    :param lag_times:
    :param add_random_strategy:
    :param add_one_over_n_strategy:
    :return:
    """
    strategy_container = []
    test_container = []
    j = 0
    data = None
    for asset_count in number_of_assets:
        if data is not None:
            del data
        data = data_container[data_container.columns[:asset_count]]
        print("final number of observations: {}".format(data.index.size))
        for method_name in covariance_methods:
            for strategy in weight_strategy_names:
                for commission_func in commission_functions:
                    for lookback in lookback_periods:
                        for lag in lag_times:
                            strategy_container.append(
                                bt.Strategy(method_name + '_' + strategy + str(lookback) + '_' + str(asset_count),
                                            [
                                                bt.algos.RunMonthly(),
                                                bt.algos.SelectAll(),
                                                get_available_strategies(lookback=lookback,
                                                                         lag=lag,
                                                                         covar_method=method_name,
                                                                         strategy_name=strategy,
                                                                         optimum_container=optimum_container,
                                                                         D_N=D_N),
                                                bt.algos.Rebalance()]))
                            test_container.append(bt.Backtest(strategy_container[j], data, commissions=commission_func))
                            j += 1

    if add_random_strategy:
        strategy_container.append(bt.Strategy('random' + str(lookback) + '_' + str(asset_count),
                                              [
                                                  bt.algos.RunMonthly(),
                                                  bt.algos.SelectAll(),
                                                  bt.algos.WeighRandomly(),
                                                  bt.algos.Rebalance()]))
        test_container.append(
            bt.Backtest(strategy_container[len(strategy_container) - 1], data, commissions=commission_func))
    if add_one_over_n_strategy:
        strategy_container.append(bt.Strategy('1/n' + str(lookback) + '_' + str(asset_count),
                                              [
                                                  bt.algos.RunMonthly(),
                                                  bt.algos.SelectAll(),
                                                  bt.algos.WeighEqually(),
                                                  bt.algos.Rebalance()]))
        test_container.append(
            bt.Backtest(strategy_container[len(strategy_container) - 1], data, commissions=commission_func))
    return strategy_container, test_container


def value_added_plot(result, covariance_methods):
    value_added = []
    number_of_covar_methods = len(covariance_methods)
    index = 0
    while index in range(len(result.stats.loc['total_return'].values)):
        current_block = result.stats.loc['total_return'].values[index:index + number_of_covar_methods]
        ref_value = current_block[0]
        for test_value in current_block:
            value_added.append((test_value - ref_value) / ref_value)
        index += number_of_covar_methods

    plt.figure(figsize=(15, 5))
    axis_fontsize = 15
    title_fontsize = 20
    plt.xticks(rotation='vertical', fontsize=axis_fontsize)
    plt.yticks(fontsize=axis_fontsize)
    plt.xlabel('Test Name', fontsize=axis_fontsize)
    plt.ylabel('Value Added', fontsize=axis_fontsize)
    plt.title('Added Value vs Test Name', fontsize=title_fontsize)
    plt.grid()
    plt.bar(result.stats.loc['total_return'].index.values, value_added)
    plt.show()
    return 0


def value_return_plot(result):
    plt.figure(figsize=(15, 5))
    axis_fontsize = 15
    title_fontsize = 20
    plt.xticks(rotation='vertical', fontsize=axis_fontsize)
    plt.yticks(fontsize=axis_fontsize)
    plt.xlabel('Test Name', fontsize=axis_fontsize)
    plt.ylabel('Return value', fontsize=axis_fontsize)
    plt.title('Return vs Test Name', fontsize=title_fontsize)
    plt.grid()
    plt.bar(result.stats.loc['total_return'].index.values, result.stats.loc['total_return'])
    plt.show()
    return 0


def show_results(result, covariance_methods, test_container, show_return_graph=True,
                 show_value_added_graph=True, show_optimum_graph=True,
                 show_optimum_vs_period_graph=False):
    # bp()
    plt.figure()
    result.plot(figsize=(15, 10), logy=True)
    # print(result.stats.loc[['total_return', 'max_drawdown',
    #                        'daily_sharpe','worst_year', 'win_year_perc']])
    if show_return_graph:
        value_return_plot(result)
    if show_value_added_graph:
        value_added_plot(result, covariance_methods)
    if show_optimum_graph:
        sorted_df = get_sorted_optimum_data(test_container)
        show_optimum_plot(sorted_df, test_container)
    if show_optimum_vs_period_graph:
        sorted_df = get_sorted_optimum_data(test_container)
        show_optimum_vs_period_plot(sorted_df, test_container)
    return 0


def get_sorted_optimum_data(test_container):
    optimum_container = pd.DataFrame(columns=OPTIMUM_CONTAINER.keys())

    optimum_container['date'] = OPTIMUM_CONTAINER['date']

    optimum_container['value'] = OPTIMUM_CONTAINER['value']
    sorted_df = optimum_container.sort_values(by='date')
    sorted_df['test_name'] = None
    sorted_df.reset_index(drop=True, inplace=True)
    for test_index, test in enumerate(test_container):
        row_index = test_index
        while row_index < sorted_df.index.size:
            sorted_df.loc[row_index, 'test_name'] = test.name
            row_index += len(test_container)
    return sorted_df


def show_optimum_plot(sorted_df, test_container):
    plt.figure(figsize=(30, 10))
    axis_fontsize = 20
    title_fontsize = 25
    plt.xticks(fontsize=axis_fontsize)
    plt.yticks(fontsize=axis_fontsize)
    plt.xlabel('Month Index', fontsize=axis_fontsize)
    plt.ylabel('Optimum', fontsize=axis_fontsize)
    plt.title('Optimum vs Month Index', fontsize=title_fontsize)
    alpha = 0.75
    for test in test_container:
        data = sorted_df[sorted_df['test_name'] == test.name].sort_values(by='date')
        t = data['date'].values.ravel()
        x = data['value'].values.ravel()
        # x = np.hstack(x)[0]
        plt.bar(np.arange(len(t)), x, alpha=alpha,
                label=test.name,
                )
    plt.legend(fontsize=axis_fontsize)
    plt.show()


def show_optimum_vs_period_plot(sorted_df, test_container):
    mean_optima_df = pd.DataFrame(columns=['test_name', 'values'])
    plt.figure(figsize=(30, 10))
    axis_fontsize = 20
    title_fontsize = 25
    plt.xticks(fontsize=axis_fontsize)
    plt.yticks(fontsize=axis_fontsize)
    for test in test_container:
        data = sorted_df[sorted_df['test_name'] == test.name].sort_values(by='date')
        # t = data['date'].values.ravel()
        x = data['value'].values.ravel()
        # x = np.hstack(x)[0]
        # bp()
        plt.hist(x, bins=50, density=True, label=test.name, alpha=0.50)
        mean_optima_df.loc[mean_optima_df.index.size] = [test.name, np.mean(x)]

    # bp()
    plt.legend()
    plt.show()
    plt.figure(figsize=(30, 10))
    axis_fontsize = 20
    title_fontsize = 25
    plt.xticks(rotation='vertical', fontsize=axis_fontsize)
    plt.yticks(fontsize=axis_fontsize)
    plt.grid()
    plt.xlabel('Test Name', fontsize=axis_fontsize)
    plt.ylabel('Average Optimum', fontsize=axis_fontsize)
    plt.title('Average Optimum vs Test Name', fontsize=title_fontsize)

    plt.plot(mean_optima_df['test_name'].values, mean_optima_df['values'].values, '-o')
    # plt.legend(fontsize=axis_fontsize)
    plt.show()


if __name__ == "__main__":
    print('working')
