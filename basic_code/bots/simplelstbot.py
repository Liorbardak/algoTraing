from .basebot import BaseBot
import pandas as pd
import numpy as np

class RandomBot(BaseBot):
    def __init__(self, name: str = 'random'):
        self._name = name

    def strategy(self, data: pd.DataFrame)->np.array:
        '''

        :param data:
        :return:
        '''
        random_vector = np.random.randint(0, 2, size=len(data))
        trade_signal = np.full(len(data), 'hold')
        trade_signal[random_vector == 0] = 'buy'
        trade_signal[random_vector == 1] = 'sell'

        return trade_signal

class DefaultBot(BaseBot):
    def __init__(self, name: str = 'default'):
        self._name = name

    def strategy(self, data: pd.DataFrame)->np.array:
        '''

        :param data:
        :return:
        '''
        # Don't buy anything - use alternative
        trade_signal = np.full(len(data), 'sell')

        return trade_signal

class SimpleBot(BaseBot):
    def __init__(self, name: str = 'simple'):
        self._name = name

    def strategy(self, data: pd.DataFrame)->np.array:
        '''

        :param data:
        :return:
        '''

        trade_signal = np.full(len(data), 'hold')
        # moving averages
        ma_150 = data['price'].rolling(window=150).mean()
        ma_20 = data['price'].rolling(window=20).mean()

        # rule #1 - hold only if stock went up in the last day 150
        rising_gap = 150
        is_rising = np.full(len(data), False)
        is_rising[rising_gap:] = ( data['price'].values[rising_gap:] - data['price'].values[:-rising_gap]) > 0
        #is_rising[rising_gap+20:] = (ma_20.values[rising_gap+20:] - ma_20.values[20:-rising_gap]) > 0


        # rule #2 - short ma is crossing long ma



        sell_ma_condition = np.full(len(data), False)
        sell_ma_condition[150:] = ma_20.values[150:] > ma_150.values[150:]*1.5

        buy_ma_condition = np.full(len(data), False)
        buy_ma_condition[150:] = ma_20.values[150:] < ma_150.values[150:]*1.0

        # trade_signal[is_rising] = 'buy'
        # trade_signal[~is_rising] = 'sell'
        # heuristics
        trade_signal[~is_rising] = 'sell'
        trade_signal[is_rising & buy_ma_condition] = 'buy'
        trade_signal[is_rising & sell_ma_condition] = 'sell'



        ## Debug
        if False:
            trade_signalnum = np.full(len(data),1 )
            trade_signalnum[trade_signal == 'buy'] = 2
            trade_signalnum[trade_signal == 'sell'] = 0

            import matplotlib
            matplotlib.use('Qt5Agg')
            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(data['price'].values)
            plt.plot(ma_20.values, label = 'ma20')
            plt.plot(ma_150.values, label = 'ma150')
            plt.plot(sell_ma_condition, label='sell_due_to_ma')
            plt.plot(is_rising ,label = 'is rising')
            plt.plot( trade_signalnum, label = 'trade signal 0 -sell , 1- hold , 2 - buy ')
            plt.legend()
            plt.title(data['name'].values[0])
            #plt.show()
        return trade_signal
