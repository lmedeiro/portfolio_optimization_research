import pandas as pd
import sys
import os
import argparse
import time
from pdb import set_trace as bp

# home_dir = os.path.expandvars("$HOME")
# app_src_dir = '/dev/repos/classes/portfolio_research_project/BackTestingSystem'
# sys.path.insert(0, home_dir + app_src_dir)
# print(home_dir + app_src_dir)



from BackTestingSystem import bt
import StrategyResources as strat_res
import pyRMT.pyRMT as rmt
import test_variables as test_vars


# Parse script arguments
def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", nargs='*', default=None,
                        help="Strategy to be used.")
    parser.add_argument("--assets", nargs='*', default=None,
                        help="Number of assets to be used.")
    parser.add_argument("--covariance", nargs='*', default=None,
                        help="Covariance methods to be used. Use comma separated values.")
    parser.add_argument("--load_test_variable_file", type=bool, default=False,
                        help="Load variables in local file called test_variables.py")
    parser.add_argument("--lookback", nargs='*', default=None,
                        help="Lookback period, in number of of days. Use comma separated values")
    return parser.parse_args(args)


def main(args):
    print(args)
    load_test_file = args.load_test_variable_file
    start_date = "2000-01-01"
    end_date = '2019-01-01'
    test_container = None
    strategy_container = None
    result = None


    data = pd.read_pickle('data.pckl')
    table_names = data.columns
    # bp()
    if load_test_file:
        weight_strategy_names = test_vars.weight_strategy_names
        covariance_methods = test_vars.covariance_methods
        number_of_assets = test_vars.number_of_assets
        lookback_periods = test_vars.lookback_periods
        indexes_to_show = test_vars.indexes_to_show
        show_return_graph = test_vars.show_return_graph
        show_value_added_graph = test_vars.show_value_added_graph
        show_optimum_graph = test_vars.show_optimum_graph
        show_optimum_vs_period_graph = test_vars.show_optimum_vs_period_graph
        save_plots = test_vars.save_plots
        test_start_time = test_vars.test_start_time

    else:
        if args.strategy is not None:
            weight_strategy_names = args.strategy
        else:
            weight_strategy_names = ['MMDP', ]

        if args.covariance is not None:
            covariance_methods = args.covariance
        else:
            covariance_methods = ['SCE', 'LWE', 'RIE']
        if args.assets is not None:
            number_of_assets = []# args.assets
            for asset in args.assets:
                number_of_assets.append(int(asset))
        else:
            number_of_assets = [10, 50, 100, 150, 200]
        if args.lookback is not None:
            lookback_days = args.lookback
            lookback_periods = []
            for day in lookback_days:
                lookback_periods.append(pd.DateOffset(days=int(day)))
        else:
            lookback_periods=[pd.DateOffset(months=6)]
        test_start_time = time.strftime('%Y-%m-%d-%H-%M-%S')
        show_return_graph = True
        show_value_added_graph = True
        show_optimum_graph = False
        show_optimum_vs_period_graph = True
        save_plots = True
        # 0: method_name
        # 1: strategy
        # 2: lookback_string
        # 3: asset_count
        indexes_to_show = {'value_added': [0, 3],
                           'equity_progression': [0, 3],
                           'optimum_graph': [0, 3],
                           'period_graph': [0, 3],
                           'return_graph': [0, 3]}

    lag_times = [pd.DateOffset(months=0)]
    # q is quantity (number of shares)
    # p is price
    #commission_fn_a = lambda q, p: q * p * 0.002
    # bp()
    commission_functions = [None]
    strat_res.OPTIMUM_CONTAINER = {'date': [], 'value': []}
    if test_container:
        del test_container
    if strategy_container:
        del strategy_container
    strategy_container, test_container = strat_res.build_test(number_of_assets,
                                                              data,
                                                              covariance_methods=covariance_methods,
                                                              weight_strategy_names=weight_strategy_names,
                                                              commission_functions=commission_functions,
                                                              lookback_periods=lookback_periods, lag_times=lag_times,
                                                              add_random_strategy=False, add_one_over_n_strategy=False,
                                                              )
    if result:
        del result

    result = bt.run(*test_container)
    strat_res.show_results(result, covariance_methods, test_container, show_return_graph=show_return_graph,
                           show_value_added_graph=show_value_added_graph,
                           show_optimum_graph=show_optimum_graph,
                           show_optimum_vs_period_graph=show_optimum_vs_period_graph,
                           save_plots=save_plots,
                           indexes_to_show=indexes_to_show,
                           test_start_time=test_start_time)


# Entry point of the script
if __name__ == "__main__":
    args = parse_args()
    main(args)
