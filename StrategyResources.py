import sys
import os
import time
import pickle

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
ACCEPTED_STRATEGIES = ['GMVP', 'MDP', 'mdp_kappa', 'MMDP', 'MMDPD', 'one_over_n']

home_dir = os.path.expandvars("$HOME")                                                                        
app_src_dir = '/dev/repos/classes/portfolio_research_project/BackTestingSystem'                               
sys.path.insert(0, home_dir + app_src_dir)                                                                    
print(home_dir + app_src_dir)  

# if 'portfolio_research_project' in cw:
#     home_dir = os.path.expandvars("$HOME")
#     app_src_dir = '/dev/repos/FinancialAnalyticsSystem/src/PortfolioManagementSystem/portfolio_research_project'
#     sys.path.insert(0, home_dir + app_src_dir)
#     from BackTestingSystem import bt
# else:
#    from src.PortfolioManagementSystem.portfolio_research_project.BackTestingSystem import bt

from BackTestingSystem import bt

def calculate_Sigma(X, method_name='SCE'):
    if "pandas" in str(type(X)):
        X = X.values
    sigma = None
    if method_name == 'SCE':
        sigma = np.cov(X, rowvar=False)
    elif method_name == 'LWE':
        sigma = sk_cov.ledoit_wolf(X)[0]
    elif method_name == 'rie' or method_name == 'RIE':
        # bp()
        if X.shape[0] <= X.shape[1]:
            # sigma = sk_cov.ledoit_wolf(X)[0]
            sigma = np.cov(X, rowvar=False)
        else:
            sigma = rmt.optimalShrinkage(X, return_covariance=True)

    elif method_name == 'clipped':
        if X.shape[0] <= X.shape[1]:
            sigma = sk_cov.ledoit_wolf(X)[0]
            # sigma = rmt.optimalShrinkage(X, return_covariance=True)
        else:
            sigma = rmt.clipped(X, return_covariance=True)
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
    for index, item in enumerate(M):
        M[index] = np.max([0, item])
    M = M.reshape([N, 1])
    sum_of_weights = np.sum(M)
    kappa = np.max([sum_of_weights, 0.001])
    M = M / kappa
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
    # TODO: Investigate the new form of this, with D being computed by the mdp_original algorithm
    # TODO: Clean this function to better reflect the definition we are posing.
    volatilities = np.sqrt(np.diag(sigma))
    volatilities = volatilities.reshape(volatilities.shape[0], 1)
    w_N = np.ones([N, 1]) * 1 / N
    V_N = w_N.T  @ volatilities
    if D_N is None:
        D_N = V_N / np.sqrt(w_N.T @ sigma @ w_N) * 1.5
        # D_N = 3
    R_N = V_N**2 / D_N**2
    x = cp.Variable(shape=(N,1))
    V = x.T @ volatilities #  / cp.quad_form(x, sigma)
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

def calculate_mmdp_based_mdp(sigma, c=0.5):
    """
    Maximum Diversified Portfolio based on a given D_N. If D_N is not given, it will be based
    on a factor of the diversified ratio of the 1/N portfolio strategy.
    :param sigma: covariance matrix
    :param min_gmvp_point: paramter that stipulates what is the minimum gmpv point at which
    to place as constraint in the QCP
    :return: x: weghts for assets, D: diversification ratio
    """
    N = sigma.shape[0]
    variance = np.diag(sigma)
    volatilities = np.sqrt(variance)
    volatilities = volatilities.reshape(volatilities.shape[0], 1)
    x = cp.Variable(shape=(N,1))
    V = x.T @ volatilities
    constraints = [x >= np.ones([N,1]) * 0,
                   np.ones([N,1]).T @ x == 1,
                   cp.quad_form(x, sigma) <= (c * np.sum(variance)) ** 2,
                   ]
    problem = cp.Problem(cp.Maximize(V),
                         constraints)
    optimum = problem.solve(qcp=True)
    x = np.array(x.value, dtype=float)
    D = V.value / np.sqrt(x.T @ sigma @ x)

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
                 bounds=(0., 1.), covar_method='LWE',
                 rf=0., lag=pd.DateOffset(days=0), optimum_container=None,
                 optimization_method='gmvp', D_N=3, mmdp_c=0.50):
        super(WeighOptimization, self).__init__()
        self.lookback = lookback
        self.lag = lag
        self.bounds = bounds
        self.covar_method = covar_method
        self.optimization_method = optimization_method
        self.optimum_container = optimum_container
        self.rf = rf
        self.D_N = D_N
        self.mmdp_c = mmdp_c

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
            if self.optimization_method == 'GMVP':
                raw_weights, optimum = calculate_gmvp(sigma)
            elif self.optimization_method == 'MDP':
                raw_weights, optimum = calculate_mdp_original(sigma)
            elif self.optimization_method == 'mdp_kappa':
                raw_weights, optimum = calculate_mdp_based_on_kappa(sigma)
            elif self.optimization_method == 'MMDP':
                # raw_weights, optimum = calculate_mdp_based_on_D(sigma, D_N=self.D_N)
                raw_weights, optimum = calculate_mmdp_based_mdp(sigma, c=self.mmdp_c)
            elif self.optimization_method == 'MMDPD':
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
                             optimum_container=None, D_N=3, mmdp_c=0.50):
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
                                 D_N=D_N, mmdp_c=mmdp_c)
    elif strategy_name == 'random':
        return bt.algos.WeighRandomly()
    else:
        raise Exception("Did not understand strategy")


def build_test(number_of_assets, data_container,
               optimum_container=None, covariance_methods=['LWE'],
               weight_strategy_names=['gmvp'], commission_functions=[None],
               lookback_periods=[pd.DateOffset(months=6)], lag_times=[pd.DateOffset(months=0)],
               add_random_strategy=False, add_one_over_n_strategy=False, D_N=3,
               mmdp_c=0.5,
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
    :param mmdp_c: parameter dictating risk aversion
    :param D_N: parameter related to portfolio diversification ratio
    :return:
    """
    strategy_container = []
    test_container = []
    j = 0
    data = None
    total_assets = len(data_container.columns)
    permutation = np.random.permutation(range(0, total_assets))
    for asset_count in number_of_assets:
        if data is not None:
            del data
        # data = data_container[data_container.columns[permutation[:asset_count]]]
        data = data_container[data_container.columns[:asset_count]]
        # bp()
        # label_dictionary = {"method_name": method_name + '_',
        #                     "strategy": strategy + '_',
        #                     "lookback_string": lookback_string + '_',
        #                     "asset_count": str(asset_count)}
        print("final number of observations: {}".format(data.index.size))
        for method_name in covariance_methods:
            for strategy in weight_strategy_names:
                for commission_func in commission_functions:
                    for lookback in lookback_periods:
                        for lag in lag_times:
                            lookback_string = str(lookback)
                            lookback_string = str.split(lookback_string)[1] # Assume the space, and separation by space.
                            lookback_string = str.replace(lookback_string, '<', '')
                            lookback_string = str.replace(lookback_string, '>', '')
                            label_details = method_name + '_'\
                                            + strategy + '_'\
                                            + lookback_string + '_'\
                                            + str(asset_count) + ' assets'
                            strategy_container.append(
                                bt.Strategy(label_details,
                                            [
                                                bt.algos.RunMonthly(),
                                                bt.algos.SelectAll(),
                                                get_available_strategies(lookback=lookback,
                                                                         lag=lag,
                                                                         covar_method=method_name,
                                                                         strategy_name=strategy,
                                                                         optimum_container=optimum_container,
                                                                         D_N=D_N,
                                                                         mmdp_c=mmdp_c),
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


def value_added_plot(result, covariance_methods, indexes_to_show, test_start_time):
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
    plt.xticks(rotation=45, fontsize=axis_fontsize)
    plt.yticks(fontsize=axis_fontsize)
    plt.xlabel('Test Name', fontsize=axis_fontsize)
    plt.ylabel('Value Added', fontsize=axis_fontsize)
    plt.title('Added Value vs Test Name', fontsize=title_fontsize)
    plt.grid()
    index_names = process_plot_label_strings(indexes=indexes_to_show,
                                             string_container=result.stats.loc['total_return'].index.values)
    plt.bar(index_names, value_added)
    result_time = test_start_time
    plt.savefig(fname='images/value_added_plot' + result_time + '.pdf', 
                format='pdf', 
                dpi=300,
                bbox_inches='tight')
    plt.show()
    return 0


def process_plot_label_strings(indexes, string_container):
    new_strings = []
    # bp()
    for item in string_container:
        data = np.array(item.split('_'))
        new_string = ''
        for s in data[indexes]:
            new_string += s + ' '
        new_strings.append(new_string)
    return new_strings


def value_return_plot(result, indexes_to_show, test_start_time):
    plt.figure(figsize=(15, 5))
    axis_fontsize = 15
    title_fontsize = 20
    plt.xticks(rotation=45, fontsize=axis_fontsize)
    plt.yticks(fontsize=axis_fontsize)
    plt.xlabel('Test Name', fontsize=axis_fontsize)
    plt.ylabel('Return value', fontsize=axis_fontsize)
    plt.title('Return vs Test Name', fontsize=title_fontsize)
    plt.grid()
    index_names = process_plot_label_strings(indexes=indexes_to_show,
                                             string_container=result.stats.loc['total_return'].index.values)
    plt.bar(index_names, result.stats.loc['total_return'])
    result_time = test_start_time
    plt.savefig(fname='images/value_return_plot' + result_time + '.pdf', 
                format='pdf', 
                dpi=300,
                bbox_inches='tight')
    plt.show()
    return 0


def show_results(result, covariance_methods, test_container, show_return_graph=True,
                 show_value_added_graph=True, show_optimum_graph=True,
                 show_optimum_vs_period_graph=False, save_plots=True, indexes_to_show=None,
                 test_start_time=time.strftime('%Y-%m-%d-%H-%M-%S')):
    if indexes_to_show is None:
        indexes_to_show = {'value_added': [0, 3],
                           'equity_progression': [0, 3],
                           'optimum_graph': [0, 2],
                           'period_graph': [0, 3],
                           'return_graph': [0, 3]}
    index_names = process_plot_label_strings(indexes=indexes_to_show['equity_progression'],
                                             string_container=result.stats.loc['total_return'].index.values)

    result.plot(figsize=(15, 10), logy=False)
    plt.legend(index_names)
    axis_fontsize = 20
    title_fontsize = 25
    plt.xticks(fontsize=axis_fontsize)
    plt.yticks(fontsize=axis_fontsize)
    plt.grid()
    plt.xlabel('Dates', fontsize=axis_fontsize)
    plt.ylabel('Percentage Returns', fontsize=axis_fontsize)
    plt.title('Equity Progression', fontsize=title_fontsize)
    result_time = test_start_time
    if save_plots:
        plt.savefig(fname='images/equity_result_plot' + result_time + '.pdf',
                    format='pdf',
                    dpi=300,
                    bbox_inches='tight')
        # pickle.dump(result,
        #             open('data/result_' + result_time + '.pckl', 'wb'))
    else:
        plt.show()
    # bp()
    if show_return_graph:
        value_return_plot(result, indexes_to_show['return_graph'], test_start_time)
    if show_value_added_graph:
        value_added_plot(result, covariance_methods, indexes_to_show['value_added'], test_start_time)
    if show_optimum_graph or show_optimum_vs_period_graph:
        sorted_df = get_sorted_optimum_data(test_container, indexes_to_show['optimum_graph'])
    if show_optimum_graph:
        show_optimum_plot(sorted_df, test_container, test_start_time)
    if show_optimum_vs_period_graph:
        show_optimum_vs_period_plot(sorted_df, test_container, test_start_time)
    return 0


def get_sorted_optimum_data(test_container, indexes_to_show):
    optimum_container = pd.DataFrame(columns=OPTIMUM_CONTAINER.keys())

    optimum_container['date'] = OPTIMUM_CONTAINER['date']

    optimum_container['value'] = OPTIMUM_CONTAINER['value']
    sorted_df = optimum_container.sort_values(by='date')
    sorted_df['test_name'] = None
    sorted_df.reset_index(drop=True, inplace=True)
    for test_index, test in enumerate(test_container):
        row_index = test_index
        while row_index < sorted_df.index.size:
            # bp()
            short_test_name = process_plot_label_strings(indexes_to_show, [test.name])
            sorted_df.loc[row_index, 'short_test_name'] = short_test_name[0]
            sorted_df.loc[row_index, 'test_name'] = test.name
            row_index += len(test_container)
    return sorted_df


def show_optimum_plot(sorted_df, test_container, test_start_time):
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
        d = 0
        # bp()
        for d in x[0].shape:
            d += 1
        if d > 1:
            x = np.hstack(x)[0]
        plt.bar(np.arange(len(t)), x, alpha=alpha,
                label=sorted_df[sorted_df['test_name'] == test.name]['short_test_name'].values[0],)
    # bp()
    plt.legend(fontsize=axis_fontsize)
    result_time = test_start_time
    plt.savefig(fname='images/optimum_plot' + result_time + '.pdf',
                format='pdf', 
                dpi=300, bbox_inches='tight')
    plt.show()


def show_optimum_vs_period_plot(sorted_df, test_container, test_start_time):
    mean_optima_df = pd.DataFrame(columns=['test_name', 'short_test_name', 'values'])
    # plt.figure(figsize=(30, 10))
    axis_fontsize = 20
    title_fontsize = 25
    plt.xticks(fontsize=axis_fontsize)
    plt.yticks(fontsize=axis_fontsize)
    for test in test_container:
        data = sorted_df[sorted_df['test_name'] == test.name].sort_values(by='date')
        # t = data['date'].values.ravel()
        x = data['value'].values.ravel()
        d = 0
        for d in x[0].shape:
            d += 1
        if d > 1:
            x = np.hstack(x)[0]
        short_test_name = sorted_df[sorted_df['test_name'] == test.name]['short_test_name'].values[0]
        # plt.hist(x, bins=50, density=True,
        #          label=short_test_name,
        #          alpha=0.50)
        mean_optima_df.loc[mean_optima_df.index.size] = [test.name, short_test_name, x]
    # TODO: Take more statistics, so to build a more complete graph, as described in the desired changes
    # TODO: Fix problem with plotting.
    # bp()
    # plt.legend()
    # plt.show()
    plt.figure(figsize=(30, 10))
    axis_fontsize = 20
    title_fontsize = 25
    plt.xticks(rotation=45, fontsize=axis_fontsize)
    plt.yticks(fontsize=axis_fontsize)
    plt.grid()
    plt.xlabel('Test Name', fontsize=axis_fontsize)
    plt.ylabel('Optimum Values', fontsize=axis_fontsize)
    plt.title('Optimum Statistics vs Test Name', fontsize=title_fontsize)
    # bp()
    plt.boxplot(mean_optima_df['values'].values, vert=True, patch_artist=True, notch=True,
                labels=mean_optima_df['short_test_name'].values)
    result_time = test_start_time
    plt.savefig(fname='images/optimum_vs_period_plot' + result_time + '.pdf', 
                format='pdf', 
                dpi=300, bbox_inches='tight')
    # plt.show()
    # mean_optima_df.to_pickle('data/' + 'mean_optima_df' + '_' + result_time + '.pckl')


if __name__ == "__main__":
    print('working')
