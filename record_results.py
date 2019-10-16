import pandas as pd
import sys
import os
import argparse

from pdb import set_trace as bp

# home_dir = os.path.expandvars("$HOME")
# app_src_dir = '/dev/repos/classes/portfolio_research_project/BackTestingSystem'
# sys.path.insert(0, home_dir + app_src_dir)
# print(home_dir + app_src_dir)



from BackTestingSystem import bt
import StrategyResources as strat_res
import pyRMT.pyRMT as rmt


# Parse script arguments
def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", nargs='*', default=None,
                        help="Strategy to be used.")
    parser.add_argument("--assets", nargs='*', default=None,
                        help="Number of assets to be used.")
    parser.add_argument("--covariance", nargs='*', default=None,
                        help="Covariance methods to be used. Use comma separated values.")
    parser.add_argument("--lookback", nargs='*', default=None,
                        help="Lookback period, in number of of days. Use comma separated values")
    return parser.parse_args(args)


def main(args):
    print(args)

    start_date = "2000-01-01"
    end_date = '2019-01-01'
    test_container = None
    strategy_container = None
    result = None


    data = pd.read_pickle('data.pckl')
    table_names = data.columns
    # bp()
    if args.strategy is not None:
        weight_strategy_names = args.strategy
    else:
        weight_strategy_names = ['mdp_D', ]

    if args.covariance is not None:
        covariance_methods = args.covariance
    else:
        covariance_methods = ['sample', 'ledoit-wolf', 'RIE']
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
                           show_value_added_graph=True,
                           show_optimum_graph=True,
                           show_optimum_vs_period_graph=True)


# Entry point of the script
if __name__ == "__main__":
    args = parse_args()
    main(args)
