import os.path
from typing import List
import pandas as pd
import matplotlib

from basic_code.utils.report_utils import HtmlReport

# matplotlib.use('Qt5Agg')
# import pylab as plt

from bots import *
from visualization.visualize_trade import visualize_trade


from basic_code.trading_simulations.tradesim import TradeSimSimple



def run_trade_sim(datadir : str ,
                  outputdir: str,
                  trade_bots : List[BaseBot] , run_this_stock_only : str =None ,
                  fix_reference_index = False ,do_report: bool = True
                 ):
    '''
    Run a simple trade simulation
    :param datadir:
    :param trade_bots:
    :param run_this_stock_only:
    :param fix_reference_index:
    :param do_report : save report per bot
    :return:
    '''
    os.makedirs(outputdir , exist_ok=True)
    # Read data
    stocks_df = pd.read_csv(os.path.join(datadir, 'all_stocks.csv'))    
    reference_index = pd.read_csv(os.path.join(datadir, 'reference_index.csv'))

    # Set the reference price to be used
    stocks_df['price'] = stocks_df['4. close']
    reference_index['price'] = reference_index['4. close']

    if fix_reference_index:
        # The alternative is to do nothing (=> the alternative price never changes)
        reference_index['price'] = 1.0

    if run_this_stock_only is not None:
        # Debug  - run only on one stock
        stocks_df = stocks_df[stocks_df['name'] == run_this_stock_only]

    #  Loop on all trade bot , simulate & report
    for trade_bot in trade_bots:

        if do_report:
            report = HtmlReport()
        else:
            report = None

        tradeSimulator = TradeSimSimple(algoBot=trade_bot)
        trade_info = tradeSimulator.run_trade_sim(stocks_df,  reference_index , report)

        visualize_trade(trade_info , stocks_df, reference_index , report)
        if do_report:
            report.to_file(os.path.join(outputdir, f"res_{trade_bot._name}.html"))
    #plt.show()








if __name__ == "__main__":
    datadir = "C:\work\Algobot\data\INCY"
    outputdir =  "C:\work\Algobot\data\TradeRes"
    run_trade_sim(datadir=datadir,outputdir=outputdir, trade_bots= [CharnyBot()] )
    # run_trade_sim(datadir=datadir, outputdir=outputdir, trade_bots=[CharnyBot()],
    #               run_this_stock_only='IMVT')







