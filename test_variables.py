import pandas as pd

# weight_strategy_names = None
weight_strategy_names = ['MMDP']

covariance_methods = ['SCE', 'LWE']

number_of_assets = [10, 20]

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
show_optimum_graph=False
show_optimum_vs_period_graph=True
save_plots=True