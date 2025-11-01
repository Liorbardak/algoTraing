import numpy as np
import pickle
import os
import pandas as pd
from tqdm import tqdm
import copy


def prepare_df(inpath = "C:/Users/dadab/projects/algotrading/results/trading_sim/all_data"):
    '''
    Repeat compliments advantage simulation
    :param inpath:
    :return:
    '''

    tickers_df, complement_df, avg_df = pickle.load(open(os.path.join(inpath, 'all_data.pickle'), "rb"))

    all_tickers = list(set(complement_df.ticker))
    tickers = all_tickers
    buy_conditions = [{'name' : 'ma150' , 'buy_condition_func': lambda df: df[(df.ma_150_slop.values > 0) & (df.ma_150.values < df.Close.values)],},
                      {'name': 'rsi', 'buy_condition_func': lambda df: df[df.rsi_14.values < df.ma_rsi_14.values * (1 - 20 / 100)],},
                      {'name': 'rsi_and_ma150', 'buy_condition_func': lambda df: df[(df.rsi_14.values < df.ma_rsi_14.values * (1 - 20 / 100)) & (df.ma_150_slop.values > 0) & (df.ma_150.values < df.Close.values)],
                       }
                      ]
    sell_conditions = [{'name' : 'ma200' , 'sell_condition_func': lambda df: df[df.ma_200.values > df.Close.values],},
                      {'name': 'rsi', 'sell_condition_func': lambda df:df[df.rsi_14.values > df.ma_rsi_14.values * (1 + 20 / 100)],},
                      ]

    info_rows = []
    for ticker in tqdm(tickers, desc="tickers", position=0, leave=True):
        df_ticker = tickers_df[tickers_df.ticker == ticker]
        if len(df_ticker) == 0:
            print(f"{ticker} has no data")
            continue
        df_ticker = copy.copy(df_ticker)
        df_ticker = df_ticker.sort_values('Date')
        df_comp_ticker = complement_df[complement_df.ticker == ticker]
        for r, comp in df_comp_ticker.iterrows():

            complements_portion_th2 = 0.5
            min_complements_th2 = 3
            min_complements_th1 = 5

            positive_analysts =  comp.number_of_analysts_with_compliments
            positive_analyst_ratio = comp.number_of_analysts_with_compliments / (comp.total_number_of_analysts + 1e-6)
            strong_consensus = positive_analyst_ratio > complements_portion_th2

            # Determine if analyst recommendations meet buy criteria
            # Two paths: either strong consensus OR high absolute number of recommendations
            complement_buy_signal = (
                    ((positive_analysts >= min_complements_th2) & strong_consensus) |
                    (positive_analysts >= min_complements_th1)
            )
            q_start_date = comp.Date + pd.Timedelta(days=2)
            next_date = df_ticker.loc[df_ticker.Date >= q_start_date, 'Date'].min()
            if next_date - q_start_date >  pd.Timedelta(days=5):
                continue
            q_start_date = next_date

            # Get market cap
            marketcap = np.nan
            mc_df = df_ticker[~np.isnan(df_ticker.MarketCap)]
            if len(mc_df):
                marketcap =  mc_df.MarketCap.values[np.argmin(np.abs(mc_df.Date - q_start_date))]

            base_row = {'ticker': ticker,
                   'marketcap': marketcap,
                   'good_comp' :complement_buy_signal,
                   'positive_analysts': positive_analysts,
                   'total_number_of_analysts': comp.total_number_of_analysts,
                   }

            df_ticker_at_this_q = df_ticker.loc[(df_ticker.Date > q_start_date ) & (
                        df_ticker.Date <= pd.Timestamp(q_start_date) + pd.Timedelta(days=90 * 1))]

            for buy_condition in buy_conditions:
                for sell_condition in sell_conditions:
                    buy_points = buy_condition['buy_condition_func'](df_ticker_at_this_q)

                    sell_date = None
                    for ri, (_ , buy_point) in enumerate(buy_points.iterrows()):

                        buy_date = buy_point.Date
                        if sell_date  and sell_date >= buy_date.tz_localize(None):
                            continue
                        df_after = df_ticker.loc[(df_ticker.Date >buy_date)]

                        if len(df_after) > 2:
                            sell_points = sell_condition['sell_condition_func'](df_after)
                            if (len(sell_points) > 0):
                                sell_date =  sell_points.Date.values[0]
                                sell_price = sell_points.Close.values[0]
                                sell_price_snp = sell_points.snp_Close.values[0]
                            else:
                                # Sell at last date
                                sell_date = df_after.Date.values[-1]
                                sell_price = df_after.Close.values[-1]
                                sell_price_snp = df_after.snp_Close.values[1]


                            row = copy.copy(base_row)
                            row.update({
                             'buy_sell_cond': f"{buy_condition['name']}_{sell_condition['name']}",
                             'buy_date': str(buy_date)[:10],
                             'sell_date' : str(sell_date)[:10],
                             'days' : (sell_date - buy_date.tz_localize(None)).days,
                             'profit' :  ((sell_price - buy_point.Close) / buy_point.Close) * 100,
                             'profit_snp': ((sell_price_snp - buy_point.snp_Close) / buy_point.snp_Close) * 100
                            })
                            info_rows.append(row)


    df = pd.DataFrame(info_rows)
    df = df.sort_values('ticker')
    df.to_csv('sim.csv', index=False)

def summarize(cfg):
        results = []
        for name, df in cfg.groupby('buy_sell_cond'):
            n = len(df)
            high_losing = np.sum(df.profit < -20) / n
            high_profit = np.sum(df.profit > 20) / n

            results.append({
                'Buy Sell Criteria': name,
                'Count': len(df),
                'Average Profit (%)': np.average(df.profit),
                'Average S&P (%)': np.average(df.profit_snp),
                'Loss >20% (%)': high_losing * 100,
                'Profit >20% (%)': high_profit * 100,
                'Average days buy-to-sell': np.round(np.average(df.days),1)
            })




        # Create DataFrame
        df = pd.DataFrame(results)

        # Print the table
        print(df.to_string(index=False))



def sim1(prepare_data = True):
    if prepare_data:
        prepare_df()
    all_dat = pd.read_csv('sim.csv')

    all_dat['good_comp']  = (all_dat.positive_analysts / (all_dat.total_number_of_analysts + 1e-3) >= 0.5) & (all_dat.positive_analysts >= 3)

    print('All Companies ')
    dat = all_dat


    dfg = dat[dat.good_comp]
    dfb = dat[~dat.good_comp]


    print('with good complements ')
    summarize(dfg)
    print('without good complements')
    summarize(dfb)




if __name__ == "__main__":
    sim1(prepare_data=True)
