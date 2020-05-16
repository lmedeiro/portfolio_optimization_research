import pandas as pd
import time

weight_strategy_names = ['MDP']
test_name = '_'
for item in weight_strategy_names:
    test_name += item
test_name += '_'
test_start_time = test_name + time.strftime('%Y-%m-%d-%H-%M-%S')

covariance_methods = ['SCE', 'LWE', 'RIE']

number_of_assets = [10, 50, 100, 150, 200]

lookback_periods = [pd.DateOffset(months=6)]

# 0: method_name
# 1: strategy
# 2: lookback_string
# 3: asset_count
indexes_to_show = {'value_added': [0, 3],
                   'equity_progression': [0, 3],
                   'optimum_graph': [0,3],
                   'period_graph': [0,3],
                   'return_graph': [0,3]}

show_return_graph=True
show_value_added_graph=True
show_optimum_graph=True
show_optimum_vs_period_graph=False
save_plots=True