import sqlalchemy as sql
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

import scipy as sci
import cvxpy as cp
from cvxopt import matrix, solvers
import sklearn.covariance as sk_cov
from pdb import set_trace as bp

home_dir = os.path.expandvars("$HOME")
app_src_dir = '/dev/repos/classes/portfolio_research_project/BackTestingSystem'
sys.path.insert(0, home_dir + app_src_dir)
print(home_dir + app_src_dir)



from BackTestingSystem import bt
import StrategyResources as strat_res
import pyRMT.pyRMT as rmt




start_date = "2000-01-01"
end_date = '2019-01-01'
test_container = None
strategy_container = None
result = None


data = pd.read_pickle('data.pckl')
table_names = data.columns


weight_strategy_names = ['mdp_original', 'mdp_D', 'one_over_n']

covariance_methods = ['ledoit-wolf']
number_of_assets = [10, 50, 100, 150, 200]
lookback_periods=[pd.DateOffset(months=6)]
lag_times = [pd.DateOffset(months=0)]
# q is quantity (number of shares)
# p is price
#commission_fn_a = lambda q, p: q * p * 0.002
commission_functions = [None]
strat_res.OPTIMUM_CONTAINER = {'date': [], 'value': []}
if test_container:
    del test_container
if strategy_container:
    del strategy_container
strategy_container, test_container = strat_res.build_test(number_of_assets,
                data,
                # optimum_container=optimum_container,
                covariance_methods=covariance_methods,
                weight_strategy_names=weight_strategy_names,
                commission_functions=commission_functions,
                lookback_periods=lookback_periods, lag_times=lag_times,
                add_random_strategy=False, add_one_over_n_strategy=False,
                )
if result:
    del result
result = bt.run(*test_container)
strat_res.show_results(result, covariance_methods, test_container, show_return_graph=True,
                       show_value_added_graph=False,
                       show_optimum_graph=True,
                       show_optimum_vs_period_graph=False)

