import pandas as pd
import numpy as np
class TradeSimSimple(object):
    def __init__(self, algoBot , name: str = 'simple trader' ):
        self._name = name
        self._algoBot = algoBot
    # def run_trade_sim(self, stocks_df: pd.DataFrame, alternative_df: pd.DataFrame):
    #     '''
    #     Simple trade - for each stock , at each time , decide if to invest  the money in the stock or in the alternative
    #     investment
    #     For now , the balance is per stock - there is no interaction between the stocks
    #     :param stocks_df : data frame with stocks values
    #     :param alternative_df : data frame with alternative investment
    #     :return:
    #     '''
    #
    #     all_info = []
    #
    #     # Init balance (per stock)
    #     number_of_stocks = {name: 0.0 for name in set(stocks_df.name)}
    #     number_of_reference_stocks = {name: 1.0 for name in set(stocks_df.name)}
    #
    #     # loop on times
    #     for ti, date in enumerate(set(stocks_df.Date)):
    #         print(ti)
    #         # Loop on stocks
    #         for stock_name, stock_df in stocks_df.groupby('name'):
    #             # Get the trading signal buy/sell/hold
    #             trade_signal = self._algoBot.strategy(stock_df)
    #
    #
    #             if (trade_signal[ti] == 'buy') & (number_of_stocks[stock_name] == 0):
    #                 # Sell all alternative & buy this stock with all your money
    #                 balance = number_of_reference_stocks[stock_name] * alternative_df[alternative_df.Date == date]['price'].values[0]
    #                 number_of_reference_stocks[stock_name] = 0
    #
    #                 number_of_stocks[stock_name] = balance / stock_df.loc[stock_df.Date == date, 'price'].values[0]
    #
    #
    #             if (trade_signal[ti] == 'sell') & (number_of_stocks != 0):
    #                 # Sell all stocks & buy alternative with all your money
    #                 balance = number_of_stocks[stock_name] * stock_df.loc[stock_df.Date == date, 'price'].values[0]
    #                 number_of_stocks[stock_name] = 0
    #
    #                 number_of_reference_stocks[stock_name] = balance / \
    #                                                alternative_df.loc[alternative_df.Date == date, 'price'].values[0]
    #
    #         # # Gather all information for this time
    #         # total_number_of_reference_stocks = 0
    #         # total_value = 0
    #         # info = {'Date' :date }
    #         # for stock_name in set(stocks_df.name):
    #         #
    #         #     total_value = total_value + number_of_reference_stocks[stock_name] * alternative_df[alternative_df.Date == date]['price'].values[0]
    #         #     total_value = total_value + number_of_stocks[stock_name] * stock_df[stock_df.Date == date]['price'].values[0]
    #         #
    #         #     info[stock_name] =  number_of_stocks[stock_name]
    #         #
    #         #     total_number_of_reference_stocks = total_number_of_reference_stocks + number_of_reference_stocks[
    #         #         stock_name]
    #         #
    #         # info['reference'] = total_number_of_reference_stocks
    #         # info['total_value'] = total_value
    #         # all_info.append(info)
    #
    #
    #     return pd.DataFrame(all_info)
    #
    # def run_trade_sim(self, stocks_df: pd.DataFrame, alternative_df: pd.DataFrame):
    #     '''
    #     Simple trade - for each stock , at each time , decide if to invest  the money in the stock or in the alternative
    #     investment
    #     :param stocks_df : data frame with stocks values
    #     :param alternative_df : data frame with alternative investment
    #     :return:
    #     '''
    #
    #     nstocks = len(set(stocks_df.name))
    #     total_balance = 0
    #
    #     # Init stock holding array
    #     number_of_reference_stocks = {name: 1.0 for name in set(stocks_df.name)}
    #
    #     for date in set(stocks_df.Date):
    #
    #     for stock_name, stock_df in stocks_df.groupby('name'):
    #         # Get the trading signal buy/sell/hold
    #         trade_signal = self._algoBot.strategy(stock_df)
    #
    #         balance = 1.0
    #         number_of_stocks = 0
    #         number_of_alternative_stocks = 0
    #         # loop on times
    #         for ti, date in enumerate(stock_df.Date):
    #             if (number_of_alternative_stocks == 0) & (number_of_stocks == 0):
    #                 # init - buy alternative with all your money
    #                 number_of_alternative_stocks = balance / \
    #                                                alternative_df.loc[alternative_df.Date == date, 'price'].values[0]
    #                 balance = 0
    #             if (trade_signal[ti] == 'buy') & (number_of_stocks == 0):
    #                 # Sell all alternative & buy this stock with all your money
    #                 balance = number_of_alternative_stocks * \
    #                           alternative_df[alternative_df.Date == date]['price'].values[0]
    #                 number_of_stocks = balance / stock_df.loc[stock_df.Date == date, 'price'].values[0]
    #                 number_of_alternative_stocks = 0
    #             if (trade_signal[ti] == 'sell') & (number_of_stocks != 0):
    #                 # Sell all stocks & buy alternative with all your money
    #                 balance = number_of_stocks * stock_df.loc[stock_df.Date == date, 'price'].values[0]
    #                 number_of_alternative_stocks = balance / \
    #                                                alternative_df.loc[alternative_df.Date == date, 'price'].values[0]
    #                 number_of_stocks = 0
    #
    #             # Store
    #             info.append({'Date': date, 'name': stock_name, 'number_of_stocks': number_of_stocks,
    #                          'number_of_alternative_stocks': number_of_alternative_stocks
    #
    #                          })
    #
    #         # At the end - sell all
    #         balance = number_of_stocks * stock_df.loc[stock_df.Date == date, 'price'].values[
    #             0] + number_of_alternative_stocks * alternative_df.loc[alternative_df.Date == date, 'price'].values[0]
    #         total_balance = total_balance + balance
    #
    #         print(f'{stock_name}  {balance}')
    #     print(f' total total {total_balance / nstocks}')
    #     return pd.DataFrame(info)

    def run_trade_sim(self, stocks_df: pd.DataFrame, alternative_df: pd.DataFrame):
        '''
        Simple trade - for each stock , at each time , decide if to invest  the money in the stock or in the alternative
        investment
        :param stocks_df : data frame with stocks values
        :param alternative_df : data frame with alternative investment
        :return:
        '''


        nstocks = len(set(stocks_df.name))
        total_balance = 0


        # Init stock holding array
        info = {}
        dates = set(stocks_df.Date)
        names = set(stocks_df.name)
        info['stocks_per_share'] = np.zeros((len(names), len(dates)))
        info['reference_stocks'] = np.zeros(len(dates), )
        info['total_value'] = np.zeros(len(dates), )
        info['Dates'] = sorted(list(dates))
        info['names'] = sorted(list(names))
        print(info['names'])

        for si,  (stock_name, stock_df) in enumerate(stocks_df.groupby('name', sort=True)):
            stock_df = stock_df.reset_index()
            print(stock_name)
            # Get the trading signal buy/sell/hold
            trade_signal = self._algoBot.strategy(stock_df)

            balance = 1.0
            number_of_stocks = 0
            number_of_alternative_stocks = 0
            # loop on times
            for ti, date in enumerate(stock_df.Date):
                if (number_of_alternative_stocks == 0) & (number_of_stocks == 0):
                    # init - buy alternative with all your money
                    number_of_alternative_stocks = balance / \
                                                   alternative_df.loc[alternative_df.Date == date, 'price'].values[0]
                    balance = 0
                if (trade_signal[ti] == 'buy') & (number_of_stocks == 0):
                    # Sell all alternative & buy this stock with all your money
                    balance = number_of_alternative_stocks * alternative_df[alternative_df.Date == date]['price'].values[0]
                    number_of_stocks = balance / stock_df.loc[stock_df.Date == date, 'price'].values[0]
                    number_of_alternative_stocks = 0
                if (trade_signal[ti] == 'sell') & (number_of_stocks != 0):
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
        return info
