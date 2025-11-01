import pylab as plt
import numpy as np
import pandas as pd
import os
import pickle
from tqdm import tqdm
import copy
import yfinance as yf
from datetime import datetime, timedelta
# At the top of your script
import sys
from pathlib import Path

# Add parent package to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Now you can use absolute imports
from basic_code.charnybot.utils.report_utils import HtmlReport
#from ..utils.report_utils import HtmlReport
import matplotlib.dates as mdates

def plot_ticker(ticker,stocks_df, complement_df , trade_df):
    '''
    Display a single ticker price date - percentage in portfolio , complements
    :param ticker:
    :param stocks_df:
    :param complement_df:
    :param trade_df:
    :return:
    '''

    # Create the plot with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12),
                                        gridspec_kw={'height_ratios': [3, 0.8, 1.2]},
                                        sharex=True)

    # Plot 1: Stock Price with Moving Averages
    ax1.plot(stocks_df.Date, stocks_df.Close, 'b-', linewidth=1, marker='o', markersize=1.5, label='Stock Price')
    ax1.plot(stocks_df.Date, stocks_df.ma_150, 'orange', linewidth=2, label='150-day MA')
    ax1.plot(stocks_df.Date, stocks_df.ma_200, 'red', linewidth=2, label='200-day MA')
    ax1.set_ylabel('Stock Price ($)', fontsize=12)
    ax1.set_title(f'{ticker} Stock Price and Analyst Compliments Analysis', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.grid(True)

    stocks_dates = np.array([str(d)[:10] for d in stocks_df.Date.values])
    buy_points = [np.where(np.array(stocks_dates) == buy_date)[0][0] for buy_date in  trade_df['buy date'].values]
    sell_points = [np.where(np.array(stocks_dates) == buy_date)[0][0] for buy_date in trade_df['maximal price date'].values]


    for buy_point, sell_point in zip(buy_points,sell_points):
        buy_Date = stocks_df.Date.values[buy_point]
        sell_Date = stocks_df.Date.values[sell_point]

        ticker_buy_price = stocks_df[stocks_df.Date.values == buy_Date].Close.values[0]
        ticker_sell_price = stocks_df[stocks_df.Date.values == sell_Date].Close.values[0]

        ax1.plot(buy_Date, ticker_buy_price,marker='o', markersize=6, color='green', linewidth=2)
        ax1.plot(sell_Date, ticker_sell_price, marker='x',markersize=6, color='red',linewidth=2)

        profit = np.round((ticker_sell_price -ticker_buy_price) / ticker_buy_price * 100,1)
        #draw profit in middle of arc connecting buy and sell points
        # Get midpoint coordinates
        mid_date =  stocks_df.Date.values[(buy_point+sell_point)//2]
        mid_price = (ticker_sell_price + ticker_buy_price) / 2

        # Add some vertical offset for better visibility
        price_range = ticker_sell_price - ticker_buy_price
        offset = abs(price_range) * 0.1  # 10% offset

        # ax1.annotate(f'{profit}%',
        #              xy=(mid_date, mid_price + offset),
        #              ha='center', va='bottom',
        #              fontsize=12, fontweight='bold',
        #              bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

    # Plot 2: RSI ...
    ax2.plot(stocks_df.Date, stocks_df.rsi_14, 'y', linewidth=2, label='RSI')
    ax2.plot(stocks_df.Date, stocks_df.ma_rsi_14, 'c', linewidth=2, label='MA_RSI')
    ax2.plot(stocks_df.Date, stocks_df.ATR_14*10, 'm', linewidth=2, label='ATR*10')

    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')
    ax2.grid(True)

    # Create stacked bar chart
    width = 15  # Width of bars in days
    analyst_dates =complement_df.Date
    ax3.bar(analyst_dates, complement_df.number_of_analysts_with_compliments, width,
            color='lightcoral', alpha=0.7, label='Number analysts with compliments')

    # Add total analysts bar (outline)
    ax3.bar(analyst_dates, complement_df.total_number_of_analysts, width,
            fill=False, edgecolor='gray', linewidth=1, label='Total Analysts')

    ax3.set_ylabel('Number of Analysts', fontsize=12)
    ax3.set_xlabel('Date', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, complement_df.total_number_of_analysts.max()+1)
    ax3.grid(True)
    # Format x-axis

    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=3, bymonthday=1))
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    #
    # Rotate x-axis labels
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Adjust layout
    plt.tight_layout()

    return fig



def charny_sim1(inpath = "C:/Users/dadab/projects/algotrading/results/trading_sim/all_data/basealgo",
                sim_type = "in_portfolio"):
    np.random.seed(12345)



    tickers_df, complement_df, avg_df = pickle.load(open(os.path.join(inpath, 'all_data.pickle'), "rb"))

    trade_hist_df = pd.read_csv(os.path.join(inpath, 'trade_simulation_results.csv'))

    all_tickers = list(set(complement_df.ticker))

    # Get snp
    start = np.min(tickers_df.Date) -timedelta(days=250)
    end =np.max(tickers_df.Date) +timedelta(days=20)


    ticker = '^GSPC'
    data = yf.download('^GSPC', start=start, end=end, auto_adjust=False)

    snp_df = pd.DataFrame()
    snp_df['Date'] = data[('Close', ticker)].index
    for k in ['High', 'Low', 'Open', 'Close', 'Volume']:
        snp_df[k] = data[(k, ticker)].values
    snp_df['ma_150'] = snp_df['Close'].rolling(window=150).mean()
    ma_diff = np.diff(snp_df['ma_150'].values)
    snp_df['ma_150_slop'] = np.hstack((ma_diff[0], ma_diff))
    snp_df['ma_50'] = snp_df['Close'].rolling(window=50).mean()
    ma_diff = np.diff(snp_df['ma_50'].values)
    snp_df['ma_50_slop'] = np.hstack((ma_diff[0], ma_diff))

    # Choose tickers
    tickers_that_were_in_portfolio = sorted([
        ticker for ticker in trade_hist_df.keys()
        if ticker in all_tickers
    ])



    if sim_type ==  "all":
        tickers = all_tickers
    elif sim_type ==  "in_portfolio":
        tickers = sorted(tickers_that_were_in_portfolio)
    elif "not_in_portfolio":
        tickers_that_were_not_in_portfolio = list(set(all_tickers) - set(tickers_that_were_in_portfolio))
        tickers = sorted(tickers_that_were_not_in_portfolio)
    else:
        assert False , "Bad sim_type"


    info_rows = []
    for ticker in tqdm(tickers, desc="tickers",position=0,  leave=True):
        df_ticker = tickers_df[tickers_df.ticker ==ticker ]
        df_ticker = copy.copy(df_ticker)

        # ADD more stuff

        # SNP
        snp_ma_150 = np.full(len(df_ticker), -1.0)
        snp_ma_50 = np.full(len(df_ticker), -1.0)
        snp_ma_150_slop = np.full(len(df_ticker), -1.0)
        snp_ma_50_slop = np.full(len(df_ticker), -1.0)
        for i, date in enumerate(df_ticker['Date'].values):
            try:
                snp_ma_150[i] = snp_df[snp_df.Date == date].ma_150.values[0]
                snp_ma_50[i] = snp_df[snp_df.Date == date].ma_50.values[0]
                snp_ma_150_slop[i] = snp_df[snp_df.Date == date].ma_150_slop.values[0]
                snp_ma_50_slop[i] = snp_df[snp_df.Date == date].ma_50_slop.values[0]
            except:
                print(f'bad date {ticker} {date}')
        df_ticker['SNP_ma_150'] = snp_ma_150
        df_ticker['SNP_ma_150_slop'] = snp_ma_150_slop
        df_ticker['SNP_ma_50'] = snp_ma_50
        df_ticker['SNP_ma_50_slop'] = snp_ma_50_slop


        # Add HV
        # Compute daily log returns
        df_ticker["log_ret"] = np.log(df_ticker["Close"]).diff()

        # Rolling (window) standard deviation of returns
        df_ticker["hv"] = df_ticker["log_ret"].rolling(10).std() * np.sqrt(252)

        df_comp_ticker = complement_df[complement_df.ticker == ticker]

        valid_market_cap_inds = np.where(~np.isnan(df_ticker.MarketCap.values))[0]

        for prec_th in[15,20,25,30,35]:

            df_comp_ticker_dates = pd.DataFrame({"date": pd.to_datetime(df_comp_ticker.date.values)})
            was_below = False
            for i in range(len(df_ticker)):
                if(str(df_ticker.Date.values[i])[:10] in df_comp_ticker.date.values):
                    # new quarter - reset the search for buying point
                    was_below = False
                    continue
                if df_ticker.rsi_14.values[i]  >  df_ticker.ma_rsi_14.values[i]:
                    # rsi crossed msa rsi - reset the search for buying point
                     was_below = False

                if not was_below and df_ticker.rsi_14.values[i]  <  df_ticker.ma_rsi_14.values[i] * (1 - prec_th / 100):
                    # Buy point
                    was_below = True
                    buying_point = i

                    # Find closest complements before and after the buying data
                    total_number_of_analysts = np.nan
                    number_of_analysts_with_compliments = np.nan
                    # TODO - change this - should be  mask = df_comp_ticker_dates["date"].values <= df_ticker.Date.values[buying_point]
                    mask = df_comp_ticker_dates["date"].values <= df_ticker.Date.values[buying_point]
                    dates_before  = (df_comp_ticker.loc[mask, "date"].values)
                    if len(dates_before) == 0:
                        continue
                    closest_complement_date_before =np.max(dates_before)
                    mask = df_comp_ticker_dates["date"].values > df_ticker.Date.values[buying_point]
                    dates_after = (df_comp_ticker.loc[mask, "date"].values)
                    if len(dates_after) == 0:
                        continue
                    closest_complement_date_after = np.min(dates_after)


                    if (type(closest_complement_date_before) == str) & (type(closest_complement_date_after) == str) :
                        # Verify that the complements date is not too far
                        buy_date = np.datetime64(df_ticker.Date.values[buying_point])
                        complements_date_before = np.datetime64(closest_complement_date_before)
                        complements_date_after = np.datetime64(closest_complement_date_after)

                        days_from_start = (buy_date.astype('datetime64[D]') - complements_date_before.astype('datetime64[D]')).astype(int)
                        days_to_end = (complements_date_after.astype('datetime64[D]') - buy_date.astype('datetime64[D]')).astype(int)

                        if (days_from_start > 0) & (days_from_start <= 100) & (days_to_end <= 100):
                            # buying date is between 2 dates of complements
                            total_number_of_analysts = df_comp_ticker[df_comp_ticker.date == closest_complement_date_before].total_number_of_analysts.values[0]
                            number_of_analysts_with_compliments = df_comp_ticker[df_comp_ticker.date == closest_complement_date_before].number_of_analysts_with_compliments.values[0]
                    if np.isnan(total_number_of_analysts):
                        # Could not find complements before & after this buying point - do not use
                        continue


                    # Find closest reported marker cap
                    market_cap = np.nan
                    if len(valid_market_cap_inds):
                        closest_ind = np.where(valid_market_cap_inds <=  buying_point)[0]
                        if len(closest_ind):
                            market_cap = df_ticker.MarketCap.values[valid_market_cap_inds[closest_ind[-1]]]

                    # Get maximal profit  until the end of the quarter
                    start_of_quarter_date = complements_date_before.astype('datetime64[D]')
                    end_of_quarter_date = complements_date_after.astype('datetime64[D]')
                    buying_point_date = df_ticker.Date.values[buying_point]

                    df_ticker_start_to_quarter_end = df_ticker[(df_ticker.Date.values >= start_of_quarter_date) & (
                            df_ticker.Date.values < end_of_quarter_date)]


                    df_ticker_buy_to_quarter_end = df_ticker[(df_ticker.Date.values >= buying_point_date) & (
                            df_ticker.Date.values < end_of_quarter_date)]

                    price_until_the_end_of_quarter = df_ticker_buy_to_quarter_end.Close.values

                    max_price_ind = price_until_the_end_of_quarter.argmax()

                    maximal_price = df_ticker_buy_to_quarter_end.Close.values[max_price_ind]

                    maximal_price_hv = df_ticker_buy_to_quarter_end.hv.values[max_price_ind]

                    maximal_price_date = df_ticker_buy_to_quarter_end.Date.values[max_price_ind]

                    days_to_maximal_price = (maximal_price_date.astype('datetime64[D]')- buy_date.astype('datetime64[D]')).astype(int)


                    # Get price difference between the day after the complements and  the buying data
                    buy_price = df_ticker.Close.values[buying_point]
                    buy_price_one_day_after_the_complements = df_ticker[(df_ticker.Date.values >= complements_date_before +  np.timedelta64(1, 'D'))].Close.values[0]

                    ATR = df_ticker.ATR_14.values[buying_point]


                    row = { 'ticker' : ticker,
                            'buy criteria' : f"rsi below rsi ma by {prec_th}%",
                            'buy date': str(buying_point_date)[:10],
                            'buy price ': buy_price,
                            'buy hv': np.max([df_ticker.hv.values[buying_point],0.1]),
                            'days_to_quarter_end': days_to_end,
                            'days_to_quarter_start': days_from_start,

                            'date of complement': closest_complement_date_before,
                            'price one day after complements' : buy_price_one_day_after_the_complements,
                            'price change from complements to buy date%' : (buy_price/buy_price_one_day_after_the_complements -1)*100,
                            'date of next complement': closest_complement_date_after,

                            'maximal price date': str(maximal_price_date)[:10],
                            'maximal price': maximal_price,
                            'maximal profit in quarter[%]' : (price_until_the_end_of_quarter.max() / df_ticker.Close.values[buying_point] -1) * 100,
                            'maximal price hv': np.max([maximal_price_hv,0.1]),
                            'days to maximal profit in quarter': days_to_maximal_price,

                            'total_number_of_analysts': total_number_of_analysts,
                            'number_of_analysts_with_compliments': number_of_analysts_with_compliments,
                            'rsi_14': df_ticker.rsi_14.values[buying_point],
                            'vix': df_ticker.vix_Close.values[buying_point],
                            'atr': (ATR / buy_price)*100,
                            'market_cap' : market_cap,
                            '(price / ma_150 -1)*100 ': (df_ticker.Close.values[buying_point] / df_ticker.ma_150.values[
                                buying_point] - 1) * 100,
                            'ma_150_slop': df_ticker.ma_150_slop.values[buying_point],
                           }

                    # Sell with ATR criteria
                    ATR_range =  np.arange(0.5,10,0.5)
                    ATRprec = ATR / buy_price
                    for profit_ATR in ATR_range:
                        target_price = buy_price*(1+profit_ATR*ATRprec)
                        sell_inds = np.where(df_ticker_buy_to_quarter_end.Close.values >= target_price)[0]
                        if len(sell_inds) > 0:
                            sell_date = df_ticker_buy_to_quarter_end.Date.values[sell_inds[0]]
                            time_diff_days = (sell_date - buy_date) // np.timedelta64(1, 'D')
                            row[f"days from buy to ATR{profit_ATR} profit"] = time_diff_days
                        else:
                            row[f"days from buy to ATR{profit_ATR} profit"] = -1

                    # Calculate hv at selling dates when  using ATR criteria
                    ATRprec = ATR / buy_price
                    for profit_ATR in ATR_range:
                        target_price = buy_price * (1 + profit_ATR * ATRprec)
                        sell_inds = np.where(df_ticker_buy_to_quarter_end.Close.values >= target_price)[0]
                        if len(sell_inds) > 0:
                            row[f"HV ATR{profit_ATR}"] = np.max([df_ticker_buy_to_quarter_end.hv.values[sell_inds[0]],0.1])
                        else:
                            row[f"HV ATR{profit_ATR}"] = -1

                    # Calculate close price to profit when using ATR criteria
                    ATRprec = ATR / buy_price
                    for profit_ATR in ATR_range:
                        target_price = buy_price * (1 + profit_ATR * ATRprec)
                        sell_inds = np.where(df_ticker_buy_to_quarter_end.Close.values >= target_price)[0]
                        if len(sell_inds) > 0:
                            row[f"Close price ATR{profit_ATR}"] = df_ticker_buy_to_quarter_end.Close.values[
                                sell_inds[0]]
                        else:
                            row[f"Close price ATR{profit_ATR}"] = -1


                    #################################################################################
                    #   Stop loss
                    #################################################################################
                    # Calculate close price to profit when using ATR criteria , add stop loss
                    ATRprec = ATR / buy_price
                    stop_loss_dict  = {}
                    for profit_ATR in ATR_range:
                        target_price = buy_price * (1 + profit_ATR * ATRprec)
                        sell_inds = np.where(df_ticker_buy_to_quarter_end.Close.values >= target_price)[0]
                        if len(sell_inds) > 0:
                            # Stop loss
                            sell_ind = sell_inds[0]
                            stop_loss =  df_ticker_buy_to_quarter_end.Close.values[sell_ind] - ATR
                            if sell_ind > 0:
                                for sind in range(sell_inds[0],len(df_ticker_buy_to_quarter_end)):
                                    if df_ticker_buy_to_quarter_end.Close.values[sind]  <  stop_loss:
                                        # Sell when price go down
                                        sell_ind = sind
                                        break
                                    # Update stop loss
                                    stop_loss = np.max([df_ticker_buy_to_quarter_end.Close.values[sind] - ATR,  stop_loss])

                            stop_loss_dict[f"Stop Loss Close price ATR{profit_ATR}"] =  df_ticker_buy_to_quarter_end.Close.values[sell_ind]
                            stop_loss_dict[f"Stop Loss  HV ATR{profit_ATR}"] = np.max(
                                [df_ticker_buy_to_quarter_end.hv.values[sell_ind], 0.1])
                            sell_date = df_ticker_buy_to_quarter_end.Date.values[sell_ind]
                            time_diff_days = (sell_date - buy_date) // np.timedelta64(1, 'D')
                            stop_loss_dict[f"Stop Loss  days from buy to ATR{profit_ATR} profit"] = time_diff_days

                        else:
                            stop_loss_dict[f"Stop Loss Close price ATR{profit_ATR}"] = -1
                            stop_loss_dict[f"Stop Loss  HV ATR{profit_ATR}"] = -1
                            stop_loss_dict[f"Stop Loss  days from buy to ATR{profit_ATR} profit"] = -1

                    # Put the stop loss data into the main data , in the order that Adi likes
                    for profit_ATR in np.arange(0.5, 10, 0.5):
                        row[f"Stop Loss  days from buy to ATR{profit_ATR} profit"] = stop_loss_dict[
                            f"Stop Loss  days from buy to ATR{profit_ATR} profit"]

                    for profit_ATR in np.arange(0.5, 10, 0.5):
                        row[f"Stop Loss  HV ATR{profit_ATR}"] = stop_loss_dict[f"Stop Loss  HV ATR{profit_ATR}"]

                    for profit_ATR in np.arange(0.5, 10, 0.5):
                        row[f"Stop Loss Close price ATR{profit_ATR}"] = stop_loss_dict[f"Stop Loss Close price ATR{profit_ATR}"]


                    #################################################################################
                    #   Stop loss - end
                    #################################################################################

                    #################################################################################
                    #   SNP
                    #################################################################################
                    row['(SPY / SPY_ma_150 -1)*100'] = (df_ticker.snp_Close.values[buying_point] / df_ticker.SNP_ma_150.values[
                        buying_point] - 1) * 100
                    row['SPY_ma_150_slop']= df_ticker.SNP_ma_150_slop.values[buying_point]

                    row['(SPY / SPY_ma_50 -1)*100']= (df_ticker.snp_Close.values[buying_point] / df_ticker.SNP_ma_50.values[
                        buying_point] - 1) * 100
                    row['SPY_ma_50_slop']= df_ticker.SNP_ma_50_slop.values[buying_point]



                    #################################################################################
                    #   Arbitrary point
                    #################################################################################
                    sell_ind = np.random.randint(0, len(df_ticker_start_to_quarter_end))
                    row[f"Arbitrary point Date"] = str(df_ticker_start_to_quarter_end.Date.values[sell_ind])[:10]
                    row[f"Arbitrary point days_to_quarter_end"] = (end_of_quarter_date.astype('datetime64[D]') -
                                                                   df_ticker_start_to_quarter_end.Date.values[
                                                                       sell_ind].astype('datetime64[D]')).astype(int)
                    row[f"Arbitrary point HV"] = np.max(
                        [df_ticker_start_to_quarter_end.hv.values[sell_ind], 0.1])

                    row[f"Arbitrary point Close price"] = df_ticker_start_to_quarter_end.Close.values[sell_ind]

                    #################################################################################
                    #  Last day of  quarter
                    #################################################################################

                    row[f"last day of quarter HV"] = np.max([df_ticker_buy_to_quarter_end.hv.values[-1],0.1])
                    row[f"last day of quarter Close price"] = df_ticker_buy_to_quarter_end.Close.values[-1]

                    row['one day after complements HV'] = df_ticker_start_to_quarter_end.hv.values[1]
                    row['one day after complements, days to end of quarter'] = (end_of_quarter_date.astype('datetime64[D]') - df_ticker_start_to_quarter_end.Date.values[1].astype('datetime64[D]')).astype(int)

                    #################################################################################
                    # SELL by rsi criteria
                    #################################################################################
                    df_sell_by_rsi = df_ticker_buy_to_quarter_end[df_ticker_buy_to_quarter_end.rsi_14 > df_ticker_buy_to_quarter_end.ma_rsi_14 * (1 + prec_th / 100)]
                    row[f"days to rsi SELL"] = -1
                    row[f"IV at rsi SELL"] = -1
                    row[f"Price at rsi SELL"] = -1
                    if len(df_sell_by_rsi) > 0:
                        row[f"days to rsi SELL"]  = pd.Timedelta(df_sell_by_rsi.Date.values[0] - buying_point_date).days
                        row[f"IV at rsi SELL"] =  df_sell_by_rsi.hv.values[0]
                        row[f"Price at rsi SELL"] = df_sell_by_rsi.Close.values[0]

                    info_rows.append(row)


    df = pd.DataFrame(info_rows)

    df.to_csv(f'charny_{sim_type}.csv')

def report_data(infile , inpath = "C:/Users/dadab/projects/algotrading/results/trading_sim/all_data"):
    tickers_df, complement_df, avg_df = pickle.load(open(os.path.join(inpath, 'all_data.pickle'), "rb"))
    cdata =  pd.read_csv(infile + '.csv')
    report = HtmlReport()
    for ticker,tdf in tqdm(cdata.groupby(['ticker'])):
        ticker = ticker[0]
        fig = plot_ticker(ticker, tickers_df[tickers_df.ticker == ticker], complement_df[complement_df.ticker == ticker] , tdf[tdf['buy criteria'] == 'rsi below rsi ma by 20%'])
        report.add_figure(ticker, fig)
        plt.close('all')
    report.to_file(infile + '.html')
if __name__ == "__main__":

    # report_data('charny_in_portfolio')
    # report_data('charny_not_in_portfolio')
    charny_sim1(sim_type="all")
    #charny_sim1(sim_type = "in_portfolio")
    #charny_sim1(sim_type="not_in_portfolio")
