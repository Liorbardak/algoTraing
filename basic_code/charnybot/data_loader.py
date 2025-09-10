import pandas as pd
import numpy as np
import os
import json
import re
import talib
from trade_policy import TradingPolicy
from config.config import ConfigManager
from utils.protofolio import Portfolio

import datetime
class FinancialDataLoaderBase:
    def __init__(self , config):
        self.config = config

    def get_stock_features(self, stock_df):
        '''
        Add features to stock financial date
        '''

        stock_df['ma_200'] = stock_df['Close'].rolling(window=200).mean()
        stock_df['ma_150'] = stock_df['Close'].rolling(window=150).mean()
        ma_150_diff = np.diff(stock_df['ma_150'].values)
        stock_df['ma_150_slop'] = np.hstack((ma_150_diff[0], ma_150_diff))
        stock_df['rsi_14'] = talib.RSI(stock_df['Close'], timeperiod=14)
        return stock_df

    def average_stock(self, all_df):
        '''
        Average stock price
        Performed using  trading simulation,  simple average is not possible , since not all tickers are present in all times
        :param all_df:
        :return: average stock price data frame
        '''

        # Check if we can perform simple averaging - all stocks appear in all dates

        # Calculate the average of all stocks by simple trading => can not preform simple average , since not all stocks are present in all times
        trader = TradingPolicy.create("MostBasic", config=ConfigManager())
        print('calculating average stock')

        avg_df_rows = []
        for date, tickers_df_per_date in all_df.sort_values('Date').groupby('Date'):

            tickers = list(set(tickers_df_per_date.ticker))

            if len(avg_df_rows) == 0:
                # initialization - buy all existing tickers in even parts

                target_value = trader.portfolio.get_total_free_value() / len(tickers)  #
                for ticker in tickers:
                    current_price = tickers_df_per_date[tickers_df_per_date.ticker == ticker].Close.values[0]
                    shares_to_buy = target_value / current_price
                    trader.portfolio.buy_stock(ticker, shares_to_buy, current_price, date)
                total_val = trader.portfolio.get_total_value()
                avg_df_rows.append(
                    {'Date': date,
                     'ticker': 'average_ticker',
                     'High': total_val,
                     'Low': total_val,
                     'Open': total_val,
                     'Close': total_val,
                     'Volume': 1,
                     }
                )
            else:
                # update prices of stocks in  portfolio
                trader.portfolio.update_prices(price_updates=tickers_df_per_date, default_index=None,
                                               update_date=date)

                # Check if there are stocks that have been added  or was removed
                remove_tickers = list(set(trader.portfolio.tickers()) - set(tickers))
                new_tickers = list(set(tickers) - set(trader.portfolio.tickers()))

                if len(remove_tickers) > 0:
                    # sell stocks that are "not there " any more
                    for ticker in remove_tickers:
                        current_price = tickers_df_per_date[tickers_df_per_date.ticker == ticker].Close.values[0]
                        shares_to_sell = trader.portfolio.positions[ticker].quantity
                        trader.portfolio.sell_stock(ticker, shares_to_sell, current_price, date)

                if len(new_tickers) > 0:
                    # buy new tickers
                    tickers_score = {new_ticker: 100 for new_ticker in new_tickers}
                    trader.buy(date, tickers_score, tickers_df_per_date)

                total_val = trader.portfolio.get_total_value()
                avg_df_rows.append(
                    {'Date': date,
                     'ticker': 'average_ticker',
                     'High': total_val,
                     'Low': total_val,
                     'Open': total_val,
                     'Close': total_val,
                     'Volume': 1,
                     }
                )
        return pd.DataFrame(avg_df_rows)

    def load_stock_data(self, tickers , min_max_dates = None , get_average_stock = False):
        """
        Load historical stock data
        """
        tickers_not_found = []
        dfs = []
        actual_min_max_dates = None
        for ticker in tickers:
            try:
                df = pd.read_csv(os.path.join(self.config.get_path("tickers_dir"),ticker, 'stockPrice.csv'))
            except:
                print(f"Could not load {ticker}")
                tickers_not_found.append(ticker)
            for kl in [k for k in df.keys() if 'Unnamed' in k]:
                df = df.drop(kl, axis=1)

            df['ticker'] = ticker
            df['Date'] = pd.to_datetime(df['Date'],utc=True)

            # Add features
            df = self.get_stock_features(df)

            if min_max_dates is not None:
                # Take only range in time
                df = df[(df.Date >= (pd.to_datetime(min_max_dates[0],utc=True) - pd.Timedelta(days=10))) &
                        (df.Date <= (pd.to_datetime(min_max_dates[1],utc=True) + pd.Timedelta(days=10)))]
            dfs.append(df)
        print('tickers_not_found'  , tickers_not_found)
        all_df = pd.concat(dfs)
        all_df.reset_index(drop=True, inplace=True)


        avg_df = None
        if get_average_stock:
            avg_df =  self.average_stock(all_df)

        return all_df , actual_min_max_dates, avg_df

    def load_complement_data(self, tickers = None , min_max_dates = None):
        """
        Load historical complements  data
        """
        if tickers is None:
            # get all tickers that their earning has been analyzed
            tickers = sorted(set([re.match(r'^([A-Za-z]+)', file).group(1).upper() for file in os.listdir(self.config.get_path("complements_dir"))]))

        dfs = []
        for ticker in tickers:
            comp_file = os.path.join(self.config.get_path("complements_dir"), ticker + '_compliment_summary.json')
            if not os.path.isfile(comp_file):
                print('No compliment data for ticker {}'.format(ticker))
                continue
            comps = json.load(open(comp_file))
            df = pd.DataFrame(comps)
            if len(df) == 0:
                print('invalid compliment data for ticker {}'.format(ticker))
                continue

            df['Date'] = pd.to_datetime(df['date'], format='ISO8601' , utc=True)

            # Format converter
            df['ticker'] = ticker
            df['number_of_analysts_comp_1'] =  df['number_of_analysts_with_quote1_compliments']
            df['number_of_analysts_comp_2'] = df['number_of_analysts_with_quote2_compliments']
            df['number_of_analysts_comp_3'] = df['number_of_analysts_with_quote3_compliments']
            df['number_of_analysts_comp'] =   df['number_of_analysts_with_compliments']

            if min_max_dates is not None:
                # Take only range in time
                df = df[(df.Date >= pd.to_datetime(min_max_dates[0],utc=True)) & (df.Date <= pd.to_datetime(min_max_dates[1],utc=True))]

            dfs.append(df)
        all_df = pd.concat(dfs)
        all_df.reset_index(drop=True, inplace=True)
        return all_df

    def load_all_data(self, tickers = None, min_max_dates = None , get_average_stock = False):
        '''
        Load historical stock data
        :param tickers:
        :param min_max_dates:
        :return:
        '''
        complement_df = self.load_complement_data(tickers, min_max_dates)
        if tickers is None:
            tickers = set(complement_df.ticker)
        stocks_df , actual_min_max_dates, avg_df  = self.load_stock_data(tickers, min_max_dates, get_average_stock = get_average_stock)


        return stocks_df , complement_df ,  avg_df

    def load_snp(self):
        snp_df, _,_ = self.load_stock_data(tickers = ['^GSPC'])
        return snp_df


if __name__ == "__main__":
    from config.config import ConfigManager

    fl = FinancialDataLoaderBase(ConfigManager())
    print(fl.load_complement_data(['ADM', 'AJG']))

    print(fl.load_stock_data(['ADM' , 'AJG' ] ,min_max_dates = ['2023-01-01', '2025-01-01'] ))
