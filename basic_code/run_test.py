import matplotlib
matplotlib.use('Qt5Agg')
import pylab as plt

from utils.preproc import (prerprocess_data, get_average_stock)
from bots.simplelstbot import (DefaultBot, RandomBot, SimpleBot)
from tradesim import TradeSimSimple

import matplotlib
matplotlib.use('Qt5Agg')  # Set backend to TkAgg
import matplotlib.pyplot as plt

def trade_sim(datadir : str ,run_this_stock_only : str =None ,fix_alternative = False , bottype = 'simple'):
    # Read data
    stocks_df = prerprocess_data(datadir)
    alternative_df = get_average_stock(stocks_df)
    # Rename
    stocks_df['price'] = stocks_df['4. close']
    alternative_df['price'] = alternative_df['4. close']
    if fix_alternative:
        # the alternative is do nothing
        alternative_df['price'] = 1.0
    if run_this_stock_only is not None:
        stocks_df = stocks_df[stocks_df['name'] == run_this_stock_only]

    # init a trade bot
    if bottype == 'simple':
        trade_bot = SimpleBot()
    else:
        trade_bot = DefaultBot()

    tradeSimulator = TradeSimSimple(algoBot = trade_bot)

    trade_info = tradeSimulator.run_trade_sim(stocks_df,  alternative_df)

    # Visualize
    for si, (stock_name, stock_df) in enumerate(stocks_df.groupby('name', sort=True)):
        stock_df = stock_df.reset_index()

        fig, ax1 = plt.subplots()

        ax2 = ax1.twinx()
        ax1.plot(trade_info['stocks_per_share'][si], 'g-')
        ax2.plot(stock_df.price / stock_df.price.values[0], 'b-')
        ax1.set_ylabel('number_of_stocks', color='g')
        ax2.set_ylabel('price', color='b')

        plt.title(f' {stock_name}')
    plt.figure()
    plt.plot(trade_info['total_value'])
    plt.title('overall balance')
    plt.show()








if __name__ == "__main__":
    datadir = "C:\work\Algobot\data\INCY"
    #trade_sim(datadir=datadir, bottype='default')
    trade_sim(datadir=datadir, fix_alternative=True)







