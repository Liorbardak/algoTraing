from typing import Dict
from .basebot import BaseBot , tradeOrder
import pandas as pd
import numpy as np




class CharnyBot(BaseBot):
    def __init__(self, name: str = 'charnybot' , params : Dict = None):
        self._name = name
        if params is not None:
            self._params = params
        else:
            self._params = {'SMA150_Slop_buy_criteria' : 0.05,
                            'SMA150_Slop_day_gap' : 50,
                            'Current_Precent_From_50SMA' : 0.05,
                            'Current_Precent_From_50SMA_to_buy': 0.005,
                            'Current_Precent_From_50SMA_to_sell': 0.05,
                            'Max_Precent_above_50SMA_Past_X_Years' : 0.7,
                            '200SMA_margin_sell' : 0.015
                            }

    def display(self, stock_name : str, stock_df: pd.DataFrame ,
                trade_signal : np.array, reference_index : pd.DataFrame = None)->"fig":
        '''

        :param stock_name:
        :param stock_df:
        :param trade_signal:
        :param reference_index:
        :return:
        '''
        import matplotlib
        matplotlib.use('Qt5Agg')
        import pylab as plt

        buy_criteria_1, buy_criteria_2, diff_to_ma50_buy_criteria, diff_to_ma50_sell_criteria, diff_to_max_ma200_sell_criteria = self.get_features(
            stock_df)
        #normalize
        stock_df['price'] = stock_df['price']/stock_df['price'].values[0] *100
        ma_200 = stock_df['price'].rolling(window=200).mean()
        ma_150 = stock_df['price'].rolling(window=150).mean()
        ma_50 = stock_df['price'].rolling(window=50).mean()


        fig = plt.figure()
        plt.plot(stock_df.price.values,label='price')
        plt.plot(ma_50.values, label='ma_50')
        plt.plot(ma_150.values, label='ma_150')
        plt.plot(ma_200.values, label='ma_200')
        plt.plot(buy_criteria_1*30, label='buy_criteria_1')
        plt.plot(buy_criteria_2*20, label='buy_criteria_2')
        plt.plot(diff_to_ma50_buy_criteria * 10, label='diff_to_ma50_buy_criteria')
        plt.plot(diff_to_ma50_sell_criteria*5, label='diff_to_ma50_sell_criteria')

        sell_points = np.where([t.order_type == 'sell' for t in trade_signal])[0]
        plt.scatter(sell_points, stock_df.price.values[sell_points], s=80, facecolors='none', edgecolors='r', label='sell')
        buy_points = np.where([t.order_type == 'buy' for t in trade_signal])[0]
        plt.scatter(buy_points, stock_df.price.values[buy_points], s=80, facecolors='none', edgecolors='b',
                    label='buy')
        plt.legend()
        plt.title(f' {stock_name}')

        return fig

    def get_features(self, stock_df: pd.DataFrame):



        # Moving averages
        ma_200 = stock_df['price'].rolling(window=200).mean()
        ma_150 = stock_df['price'].rolling(window=150).mean()
        ma_50 = stock_df['price'].rolling(window=20).mean()

        # ma_150_Slop_buy_criteria
        dt = self._params['SMA150_Slop_day_gap']
        ma_150_Slop_buy_criteria = np.full(len(stock_df), False)
        ma_150_Slop_buy_criteria[dt:] = (ma_150.values[dt:] / ma_150.values[:-dt]) > (1. +  self._params['SMA150_Slop_buy_criteria'])
        ma_150_Slop_buy_criteria[:150] = False
        # price is above ma_150
        price_is_above_ma_150_buy_criteria = np.full(len(stock_df), False)
        price_is_above_ma_150_buy_criteria[150:] = stock_df['price'].values[150:] > ma_150.values[150:]
        # ma_50 is above ma_150
        ma_50_ma_150_buy_criteria = np.full(len(stock_df), False)
        ma_50_ma_150_buy_criteria[150:] = ma_50.values[150:] > ma_150.values[150:]

        buy_criteria_1 = np.logical_and(np.logical_and(ma_50_ma_150_buy_criteria , price_is_above_ma_150_buy_criteria) , ma_150_Slop_buy_criteria)

        ratio_to_ma50 =    stock_df['price'].values / ma_50.values

        # price is above ma50 , but not by too much
        diff_to_ma50_buy_criteria = np.full(len(stock_df), False)
        diff_to_ma50_buy_criteria[150:] =  (ratio_to_ma50[150:] > 1) & (ratio_to_ma50[150:] < (1+ self._params['Current_Precent_From_50SMA_to_buy']))

        # price is below the maximal values of ma 50 in the past
        isnan50 = np.isnan(ma_50)
        ma_50[isnan50 ] = 1e-3

        diff_to_max_ma50_buy_criteria = stock_df['price'].values / ma_50.rolling(window=200).max()   < (1 +  self._params['Max_Precent_above_50SMA_Past_X_Years'])

       # buy_criteria_2 = diff_to_ma50_buy_criteria & diff_to_max_ma50_buy_criteria
        buy_criteria_2 =  diff_to_max_ma50_buy_criteria.values

        ratio_to_ma200 = stock_df['price'].values / ma_200.values

        # Sell condition
        # price is much higher than ma50
        diff_to_ma50_sell_criteria = np.full(len(stock_df), False)
        diff_to_ma50_sell_criteria[150:] =  ratio_to_ma50[150:] >   (1+ self._params['Current_Precent_From_50SMA_to_sell'])
        # price is lower  than ma200
        diff_to_max_ma200_sell_criteria = np.full(len(stock_df), False)
        diff_to_max_ma200_sell_criteria[200:] = ratio_to_ma200[200:] < (1 - self._params['200SMA_margin_sell'])

        # import matplotlib
        # matplotlib.use('Qt5Agg')
        # import pylab as plt
        # fig = plt.figure()
        # plt.plot(stock_df.price.values, label='price')
        # plt.plot(ma_50.values, label='ma_50')
        # plt.plot(ma_150.values, label='ma_150')
        # plt.plot(ma_200.values, label='ma_200')
        # plt.plot(buy_criteria_1 * 30, label='buy_criteria_1')
        # plt.plot(buy_criteria_2 * 20, label='buy_criteria_2')
        # plt.plot(diff_to_ma50_buy_criteria * 10, label='diff_to_ma50_buy_criteria')
        # plt.plot(diff_to_ma50_sell_criteria * 5, label='diff_to_ma50_sell_criteria')
        #
        # plt.legend()
        # plt.show()


        return  buy_criteria_1, buy_criteria_2,diff_to_ma50_buy_criteria, diff_to_ma50_sell_criteria, diff_to_max_ma200_sell_criteria





    def strategy(self, stock_df: pd.DataFrame)->np.array:
        '''
        Threshold Criteria
            Stocks that passed preliminary filtering will be evaluated against 2 Threshold
            Criterions as follows:
             Threshold Criteria 1
            Stock price is above the 150-days Simple Moving Average (SMA) in an
            Uptrend (150SMA positive slope) for a predefined period.
            o [Current_Stock_Price] is above the 150-day moving average
            [Current_150SMA] and the consecutive number of days above the 150-
            day moving average from today and backward [Days_above_150SMA]
            greater than: {Days_above_150SMA_buy_criteria}.
             Default: 10 days.
            o Slope of the 150-day moving average [Current_150SMA_Slop] greater
            than: {150SMA_Slop_buy_criteria}.
             Default: 0.
            Comment: It is also possible to consider whether the 50-day moving
            average is above the 150-day moving average (usually correlated with
            an upward slope of the 150-day moving average).

             Threshold Criteria 2
            Stock price not too far from the 50-day Moving Average, the purpose of this
            criteria is to verify the stock is not currently expensive and may require few
            days to consolidate
            o Percentage distance from the 50-day moving average
            [Current_Precent_From_50SMA] – positive and smaller than
            [Max_Precent_above_50SMA_Past_X_Years] – maximum distance above the
            150-day moving average in the past
            {Max_Precent_above_50SMA_Years_Periode} years.
             Default: 5 years.

        Buying and Selling Daily
             Selling a Stock conditions:
            If the stock price drops below the 200SMA minus a margin:
            &lt;[Current_200SMA]-{200SMA_margin_sell}[Current_Stock_Price]
            o Default Margin: 1.5% below the 200-day SMA.
            o Future Option: Analyze historical stock behavior relative to the 200-
            day SMA:
             Each time a stock deviates from the 150-day moving average by more than a
                &quot;normal&quot; distance, sell a portion of the holdings
                o {Take_Profit_Sum}
                o Default: Sell 1/3 of the holding.

             Re-Purchase When Returning Close to the 150-Day Moving Average:
            If a stock previously sold for profit returns to within a distance smaller than
            {Add_to_Stock_distance_above_150SMA} of the 150SAM, buy back amount of
            {Add_to_Stock_Sum}
            o Default Distance: Below 5% above the SMA150.
            o Default Amount: 1/3 of the holding.


        :param data:
        :return:
        '''

        buy_criteria_1, buy_criteria_2,diff_to_ma50_buy_criteria, diff_to_ma50_sell_criteria, diff_to_max_ma200_sell_criteria  = self.get_features(stock_df)


        # Heuristics
        trade_criteria = np.full(len(stock_df), 0)
        #trade_criteria[ (buy_criteria_1 & buy_criteria_2)] = 1 # buy
        trade_criteria[~(buy_criteria_1 & buy_criteria_2) | ( diff_to_ma50_sell_criteria | diff_to_max_ma200_sell_criteria)] = -1  # sell
        trade_criteria[(buy_criteria_1 & buy_criteria_2) & diff_to_ma50_buy_criteria] = 1  # buy
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
