from typing import Dict
from .basebot import BaseBot , tradeOrder
import pandas as pd
import numpy as np
# import matplotlib
# matplotlib.use('Qt5Agg')
import pylab as plt

class DefaultBot(BaseBot):
    def __init__(self, name: str = 'default' , params : Dict = None):
        self._name = name
        self._params = params

    def strategy(self, data: pd.DataFrame)->np.array:
        '''

        :param data:
        :return:
        '''
        # Don't buy anything - use alternative
        trade_signal = np.full(len(data), tradeOrder('sell'))

        return trade_signal
    def display(self, stock_name : str, stock_df: pd.DataFrame ,
                trade_signal : np.array, reference_index : pd.DataFrame = None)->"fig":
        fig = plt.figure()
        plt.plot(stock_df.price.values, label='price')

        return fig


class SimpleBot(BaseBot):
    def __init__(self, name: str = 'simple' , params : Dict = None):
        self._name = name
        self._params = params
    def display(self, stock_name : str, stock_df: pd.DataFrame ,
                trade_signal : np.array, reference_index : pd.DataFrame = None)->"fig":
        '''

        :param stock_name:
        :param stock_df:
        :param trade_signal:
        :param reference_index:
        :return:
        '''
        ma_150 = stock_df['price'].rolling(window=150).mean()
        ma_20 = stock_df['price'].rolling(window=20).mean()
        rising_gap = 150
        is_rising = np.full(len(stock_df), False)
        is_rising[rising_gap:] = ( stock_df['price'].values[rising_gap:] - stock_df['price'].values[:-rising_gap]) > 0

        fig = plt.figure()
        plt.plot(stock_df.price.values,label='price')
        plt.plot(ma_150.values, label='ma_150')
        plt.plot(ma_20.values, label='ma_20')
        sell_points = np.where([t.order_type == 'sell' for t in trade_signal])[0]
        plt.scatter(sell_points, stock_df.price.values[sell_points], s=80, facecolors='none', edgecolors='r', label='sell')
        buy_points = np.where([t.order_type == 'buy' for t in trade_signal])[0]
        plt.scatter(buy_points, stock_df.price.values[buy_points], s=80, facecolors='none', edgecolors='b',
                    label='buy')
        plt.legend()
        plt.title(f' {stock_name}')

        return fig



    def strategy(self, stock_df: pd.DataFrame)->np.array:
        '''

        :param data:
        :return:
        '''


        # moving averages
        ma_150 = stock_df['price'].rolling(window=150).mean()
        ma_20 = stock_df['price'].rolling(window=20).mean()

        # rule #1 - hold only if stock went up in the last day 150
        rising_gap = 150
        is_rising = np.full(len(stock_df), False)
        is_rising[rising_gap:] = ( stock_df['price'].values[rising_gap:] - stock_df['price'].values[:-rising_gap]) > 0



        # rule #2 - short ma is crossing long ma

        sell_ma_condition = np.full(len(stock_df), False)
        sell_ma_condition[150:] = ma_20.values[150:] > ma_150.values[150:]*1.5

        buy_ma_condition = np.full(len(stock_df), False)
        buy_ma_condition[150:] = ma_20.values[150:] < ma_150.values[150:]*1.0

        # heuristics

        trade_criteria = np.full(len(stock_df), 0)
        trade_criteria[~is_rising] = -1 # sell
        trade_criteria[is_rising & buy_ma_condition] = 1 # buy
        trade_criteria[is_rising & sell_ma_condition] = -1 #sell

        # convert to single action of sell/buy all
        nstocks = 0
        trade_signal = np.full(len(stock_df), tradeOrder('hold'))
        for t in np.arange(len(stock_df)):
            if (nstocks == 0) & (trade_criteria[t] == 1):
                trade_signal[t] = tradeOrder('buy')
                nstocks = 100
            elif  (nstocks != 0) & (trade_criteria[t] == -1):
                trade_signal[t] = tradeOrder('sell')
                nstocks = 0


        return trade_signal
