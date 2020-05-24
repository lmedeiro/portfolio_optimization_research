import pandas as pd
import time

weight_strategy_names = ['MMDP', 'MDP', 'MMDPD', 'one_over_n'] # MMDP
test_name = '_'
for item in weight_strategy_names:
    test_name += item
test_name += '_'
test_start_time = test_name + time.strftime('%Y-%m-%d-%H-%M-%S')

covariance_methods = ['LWE']

number_of_assets = [100]

lookback_periods = [pd.DateOffset(months=6)]

# 0: method_name
# 1: strategy
# 2: lookback_string
# 3: asset_count
indexes_to_show = {'value_added': [1, 3],
                   'equity_progression': [1, 3],
                   'optimum_graph': [1,3],
                   'period_graph': [0,3],
                   'return_graph': [1,3]}

show_return_graph=True
show_value_added_graph=False
show_optimum_graph=False
show_optimum_vs_period_graph=False
save_plots=True