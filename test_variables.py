import pandas as pd
import time

weight_strategy_names = ['MMDPC'] # ['MMDP', 'MMDPC', 'MDP', 'GMVP']
test_name = '_'
for item in weight_strategy_names:
    test_name += item
test_name += '_'
test_start_time = test_name + time.strftime('%Y-%m-%d-%H-%M-%S')

covariance_methods = ['LWE'] # ['RIE', 'LWE', 'SCE']

number_of_assets = [150] # [10, 50, 100, 150, 200]

lookback_periods = [pd.DateOffset(months=1), pd.DateOffset(months=3), pd.DateOffset(months=6), pd.DateOffset(months=9),
                    pd.DateOffset(months=12)]

# [pd.DateOffset(months=1), pd.DateOffset(months=3), pd.DateOffset(months=6), pd.DateOffset(months=9),
#                     pd.DateOffset(months=12)]
mmdp_c = [1.5] # [0, 0.50, 1, 1.5, ...,  inf)

data_metric_key = ['optimum'] # ['optimum', 'risk']
# 0: weight_strategy_names
# 1: lookback_string
# 2: covariance_methods
# 3: asset_count
# 4: mmdp_c
indexes_to_show = {'value_added': [1],
                   'equity_progression': [1],
                   'optimum_graph': [1],
                   'period_graph': [1],
                   'return_graph': [1],
                   'weight_graph': [1]}

show_return_graph=True
show_value_added_graph=True
show_optimum_graph=True
show_optimum_vs_period_graph=True
show_weights_plot = True
save_plots=True
