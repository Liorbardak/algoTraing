from typing import Dict
import pandas as pd
import numpy as np
class TradeSimSimple(object):
    def __init__(self, algoBot : "BaseBot" , name: str = 'simple trader' ):
        self._name = name
        self._algoBot = algoBot

    def run_trade_sim(self, stocks_df: pd.DataFrame, alternative_df: pd.DataFrame , report : "HtmlReport" = None)->Dict:
        '''
        Simple trade - for each stock , at each time , decide if to invest  the money in the stock or in the alternative
        investment
        :param stocks_df : data frame with stocks values
        :param alternative_df : data frame with alternative investment
        :return:
        '''

        # Init stock holding array
        info = {}
        dates = set(stocks_df.Date)
        names = set(stocks_df.name)
        info['stocks_per_share'] = np.zeros((len(names), len(dates)))
        info['reference_stocks'] = np.zeros(len(dates), )
        info['total_value'] = np.zeros(len(dates), )
        info['Dates'] = sorted(list(dates))
        info['names'] = sorted(list(names))
        print(f"Stocks to simulate with bot {self._algoBot._name}")
        print(info['names'])

        # loop on all stock , run bot strategy  per stock
        for si,  (stock_name, stock_df) in enumerate(stocks_df.groupby('name', sort=True)):
            stock_df = stock_df.reset_index()
            print(stock_name)
            # Get the trading signal buy/sell/hold for all times
            trade_signal = self._algoBot.strategy(stock_df)

            balance = 1.0
            number_of_stocks = 0
            number_of_alternative_stocks = 0
            # loop on times ,apply the tra
            for ti, date in enumerate(stock_df.Date):
                if (number_of_alternative_stocks == 0) & (number_of_stocks == 0):
                    # init - buy alternative with all your money
                    number_of_alternative_stocks = balance / \
                                                   alternative_df.loc[alternative_df.Date == date, 'price'].values[0]
                    balance = 0
                if (trade_signal[ti].order_type == 'buy') & (number_of_stocks == 0):
                    # Sell all alternative & buy this stock with all your money
                    balance = number_of_alternative_stocks * alternative_df[alternative_df.Date == date]['price'].values[0]
                    number_of_stocks = balance / stock_df.loc[stock_df.Date == date, 'price'].values[0]
                    number_of_alternative_stocks = 0
                if (trade_signal[ti].order_type == 'sell') & (number_of_stocks != 0):
                    # Sell all stocks & buy alternative with all your money
                    balance = number_of_stocks * stock_df.loc[stock_df.Date == date, 'price'].values[0]
                    number_of_alternative_stocks = balance / \
                                                   alternative_df.loc[alternative_df.Date == date, 'price'].values[0]
                    number_of_stocks = 0

                # Store data
                info['stocks_per_share'][si, ti] = number_of_stocks
                info['reference_stocks'][ti] = number_of_alternative_stocks
                info['total_value'][ti] += number_of_stocks * stock_df.loc[stock_df.Date == date, 'price'].values[0]
                info['total_value'][ti] += number_of_alternative_stocks * \
                                                   alternative_df[alternative_df.Date == date]['price'].values[0]

            if (report is not None):
                fig = self._algoBot.display(stock_name, stock_df , trade_signal)
                if fig is not None:
                    report.add_figure(stock_name, fig)

        return info
