import numpy as np
import pickle
import os
import pandas as pd
from tqdm import tqdm
import copy

def sell_stock(buy_points , df_ticker , name):
    buy_date = np.nan
    buy_price = np.nan
    buy_price_snp = np.nan

    sell_rsi_date = np.nan
    sell_rsi_price = np.nan
    sell_rsi_price_snp = np.nan

    sell_below_ma200_date = np.nan
    sell_below_ma200_price = np.nan
    sell_below_ma200_price_snp = np.nan

    sell_below_ma200_or_rsi_date = np.nan
    sell_below_ma200_or_rsi_price = np.nan
    sell_below_ma200_or_rsi_price_snp = np.nan

    sell_q_end_date =  np.nan
    sell_q_end_price =  np.nan
    sell_q_end_price_snp =  np.nan

    if (len(buy_points)):
        buy_date = buy_points.Date.values[0]
        buy_price = buy_points.Close.values[0]
        buy_price_snp = buy_points.snp_Close.values[0]
        df_after = df_ticker.loc[(df_ticker.Date > pd.Timestamp(buy_date, tz='UTC'))]
        if len(df_after) > 20:
            sell_rsi_date = df_after.Date.values[-1]
            sell_rsi_price = df_after.Close.values[-1]
            sell_rsi_price_snp = df_after.snp_Close.values[-1]

            sell_below_ma200_date = df_after.Date.values[-1]
            sell_below_ma200_price = df_after.Close.values[-1]
            sell_below_ma200_price_snp = df_after.snp_Close.values[-1]

            # Sell by RSI
            prec_rsi_th = 20
            sell_points = df_after[
                df_after.rsi_14.values > df_after.ma_rsi_14.values * (
                        1 + prec_rsi_th / 100)]
            if len(sell_points):
                sell_rsi_date = sell_points.Date.values[0]
                sell_rsi_price = sell_points.Close.values[0]
                sell_rsi_price_snp = sell_points.snp_Close.values[0]

            # Sell by price below ma200
            sell_points = df_after[df_after.ma_200.values > df_after.Close.values]
            if len(sell_points):
                sell_below_ma200_date = sell_points.Date.values[0]
                sell_below_ma200_price = sell_points.Close.values[0]
                sell_below_ma200_price_snp = sell_points.snp_Close.values[0]

            # Sell by price below ma200 or rsi
            sell_points = df_after[(df_after.ma_200.values > df_after.Close.values) |  (df_after.rsi_14.values > df_after.ma_rsi_14.values * (
                        1 + prec_rsi_th / 100)) ]
            if len(sell_points):
                sell_below_ma200_or_rsi_date = sell_points.Date.values[0]
                sell_below_ma200_or_rsi_price = sell_points.Close.values[0]
                sell_below_ma200_or_rsi_price_snp = sell_points.snp_Close.values[0]


    r = {f'buy_{name}_date': buy_date,
            f'buy_{name}_price': buy_price,
            f'buy_{name}_price_snp': buy_price_snp,
            f'buy_{name}_sell_rsi_date': sell_rsi_date,
            f'buy_{name}_sell_rsi_price': sell_rsi_price,
            f'buy_{name}_sell_rsi_price_snp': sell_rsi_price_snp,

            f'buy_{name}_sell_below_ma200_date': sell_below_ma200_date,
            f'buy_{name}_sell_below_ma200_price': sell_below_ma200_price,
            f'buy_{name}_sell_below_ma200_price_snp': sell_below_ma200_price_snp,

            f'buy_{name}_sell_below_ma200_or_rsi_date': sell_below_ma200_or_rsi_date,
            f'buy_{name}_sell_below_ma200_or_rsi_price': sell_below_ma200_or_rsi_price,
            f'buy_{name}_sell_below_ma200_or_rsi_price_snp': sell_below_ma200_or_rsi_price_snp,

            f'buy_{name}_sell_q_end_date': sell_q_end_date,
            f'buy_{name}_sell_q_end_price': sell_q_end_price,
            f'buy_{name}_sell_q_end_price_snp':  sell_q_end_price_snp
            }
    return r


def prepare_df(inpath = "C:/Users/dadab/projects/algotrading/results/trading_sim/all_data"):
    '''
    Repeat compliments advantage simulation
    :param inpath:
    :return:
    '''

    tickers_df, complement_df, avg_df = pickle.load(open(os.path.join(inpath, 'all_data.pickle'), "rb"))

    all_tickers = list(set(complement_df.ticker))
    tickers = all_tickers
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
            buy_price = df_ticker[df_ticker.Date== q_start_date].Close.values[0]
            buy_price_snp = df_ticker[df_ticker.Date == q_start_date].snp_Close.values[0]


            # Get market cap
            marketcap = np.nan
            mc_df = df_ticker[~np.isnan(df_ticker.MarketCap)]
            if len(mc_df):
                marketcap =  mc_df.MarketCap.values[np.argmin(np.abs(mc_df.Date - q_start_date))]

            row = {'ticker': ticker,
                   'marketcap': marketcap,
                   'buy_at_q_start_date': q_start_date,
                   'buy_at_q_start_price' : buy_price,
                   'buy_at_q_start_snp_price': buy_price_snp,

                   'good_comp' :complement_buy_signal,
                   'positive_analysts': positive_analysts,
                   'total_number_of_analysts': comp.total_number_of_analysts,
                   }
            for sell_after in [30*2, 30*4,30*6]:
                after_q = df_ticker[df_ticker.Date >= pd.Timestamp(q_start_date) + pd.Timedelta(days=sell_after)]
                after_q = after_q.sort_values('Date')
                if len(after_q):
                    row.update({
                        f"sell_price_after_{sell_after}_end": after_q.Close.values[0],
                        f"sell_price_after_{sell_after}_end_snp": after_q.snp_Close.values[0],
                        f"sell_price_after_{sell_after}_end_date": after_q.Date.values[0],
                    })
                else:
                    row.update({
                        f"sell_price_after_{sell_after}_end": np.nan,
                        f"sell_price_after_{sell_after}_end_snp":  np.nan,
                        f"sell_price_after_{sell_after}_end_date":  np.nan,
                    })




            df_ticker_at_this_q = df_ticker.loc[(df_ticker.Date > q_start_date) & (df_ticker.Date <=  pd.Timestamp(q_start_date) + pd.Timedelta(days=90 * 1))]


            #######################################################################################
            # Buy by ma150  condition
            #######################################################################################

            buy_points = df_ticker_at_this_q[(df_ticker_at_this_q.ma_150_slop.values > 0) & (df_ticker_at_this_q.ma_150.values < df_ticker_at_this_q.Close.values) ]
            r = sell_stock(buy_points, df_ticker,'ma150')
            row.update(r)

            #######################################################################################
            # Buy by RSI  condition
            #######################################################################################
            prec_rsi_th = 20
            buy_points = df_ticker_at_this_q[
                df_ticker_at_this_q.rsi_14.values < df_ticker_at_this_q.ma_rsi_14.values * (1 - prec_rsi_th / 100)]
            r = sell_stock(buy_points, df_ticker, 'rsi')
            row.update(r)


            #######################################################################################
            # Buy by RSI + MA150  condition
            #######################################################################################
            prec_rsi_th = 20
            buy_points = df_ticker_at_this_q[
                df_ticker_at_this_q.rsi_14.values < df_ticker_at_this_q.ma_rsi_14.values * (1 - prec_rsi_th / 100)]
            r = sell_stock(buy_points, df_ticker, 'rsi')
            row.update(r)


            info_rows.append(row)
    df = pd.DataFrame(info_rows)
    df = df.sort_values('ticker')
    df.to_csv('info_rows.csv', index=False)
def summarize(cfg):
        results = []

        for sell_after in [30*2, 30*4,30*6]:
            profit = cfg[f"sell_price_after_{sell_after}_end"].values / cfg[f"buy_at_q_start_price"].values
            profit_snp = cfg[f"sell_price_after_{sell_after}_end_snp"].values / cfg[f"buy_at_q_start_snp_price"].values

            days_diff = pd.to_datetime(cfg[f"sell_price_after_{sell_after}_end_date"]).dt.tz_localize(None) - pd.to_datetime(
                cfg['buy_at_q_start_date']).dt.tz_localize(None)


            is_valid = ~np.isnan(profit)
            profit_snp = profit_snp[is_valid]
            profit = profit[is_valid]
            days_diff = days_diff[is_valid]
            average_days = days_diff.dt.days.mean()

            n = len(profit)
            high_losing = np.sum(profit < 0.8) / n
            high_profit = np.sum(profit > 1.2) / n

            results.append({
                'Buy Criteria': 'q start',
                'Sell Criteria': f'sell after {sell_after} days',
                'Count': n,
                'Average Profit (%)': (np.average(profit) - 1) * 100,
                'Average S&P (%)': (np.average(profit_snp) - 1) * 100,
                'Loss >20% (%)': high_losing * 100,
                'Profit >20% (%)': high_profit * 100,
                'Average days buy-to-sell': np.round(average_days,1)
            })


        buy_criteria = ['ma150', 'rsi']
        sell_criteria = ['below_ma200', 'rsi', 'below_ma200_or_rsi']

        for bc in buy_criteria:
            for sc in sell_criteria:
                profit = cfg[f"buy_{bc}_sell_{sc}_price"].values / cfg[f"buy_{bc}_price"].values
                profit_snp = cfg[f"buy_{bc}_sell_{sc}_price_snp"].values / cfg[f"buy_{bc}_price_snp"].values
                days_diff = pd.to_datetime(cfg[f'buy_{bc}_sell_{sc}_date']).dt.tz_localize(None) - pd.to_datetime(
                    cfg[f"buy_{bc}_date"]).dt.tz_localize(None)

                is_valid = ~np.isnan(profit)
                profit_snp = profit_snp[is_valid]
                profit = profit[is_valid]
                days_diff = days_diff[is_valid]
                average_days = days_diff.dt.days.mean()

                n = len(profit)
                high_losing = np.sum(profit < 0.8) / n
                high_profit = np.sum(profit > 1.2) / n

                # Store results in dictionary
                results.append({
                    'Buy Criteria': bc,
                    'Sell Criteria': sc,
                    'Count': n,
                    'Average Profit (%)': (np.average(profit) - 1) * 100,
                    'Average S&P (%)': (np.average(profit_snp) - 1) * 100,
                    'Loss >20% (%)': high_losing * 100,
                    'Profit >20% (%)': high_profit * 100,
                    'Average days buy-to-sell': np.round(average_days, 1)
                })

        # Create DataFrame
        df = pd.DataFrame(results)

        # Print the table
        print(df.to_string(index=False))



def sim1(prepare_data = True):
    if prepare_data:
        prepare_df()
    all_dat = pd.read_csv('info_rows.csv')

    all_dat['good_comp']  = (all_dat.positive_analysts / (all_dat.total_number_of_analysts + 1e-3) >= 0.5) & (all_dat.positive_analysts >= 3)

    print('All Companies ')
    dat = all_dat


    dfg = dat[dat.good_comp]
    dfb = dat[~dat.good_comp]


    print('with good complements ')
    summarize(dfg)
    print('without good complements')
    summarize(dfb)

    #
    # print('Small Cap Companies ')
    # dat = all_dat[all_dat.marketcap < 1e10]
    #
    #
    # dfg = dat[dat.good_comp]
    # dfb = dat[~dat.good_comp]
    #
    #
    # print('with good complements ')
    # summarize(dfg)
    # print('without good complements')
    # summarize(dfb)
    #
    # print('Big Cap Companies ')
    # dat = all_dat[all_dat.marketcap >= 1e10]
    #
    # dfg = dat[dat.good_comp]
    # dfb = dat[~dat.good_comp]
    #
    # print('with good complements ')
    # summarize(dfg)
    # print('without good complements')
    # summarize(dfb)


if __name__ == "__main__":
    sim1(prepare_data=True)
