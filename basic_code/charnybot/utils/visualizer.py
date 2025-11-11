import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.dates as mdates

def plot_overall( snp_df , trade_hist_df , avg_df = None):
    '''
    General plot of np vs bot
    '''
    fig, (ax1, ax2 , ax3) = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    ax1.plot( snp_df.Date ,snp_df.Close / snp_df.Close.values[0]*100 , label= 'snp')
    ax1.plot(trade_hist_df.Date, trade_hist_df.total_value.values/ trade_hist_df.total_value.values[0]*100 , label='trade')
    if avg_df is not None:
        ax1.plot(avg_df.Date, avg_df.Close /avg_df.Close.values[0]*100 ,
                 label='average stock')

    ax1.set_ylabel('Close Price')
    ax1.legend()
    ax1.grid(True)
    ax2.plot(trade_hist_df.Date, trade_hist_df.n_ticker_in_protofolio)

    ax2.set_ylabel('Number of stocks in portfolio ')
    ax2.set_xlabel('Date')
    ax2.grid(True)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    ax3.set_ylabel('Portion of s&p in portfolio ')
    ax3.plot(trade_hist_df.Date, trade_hist_df.default_index / trade_hist_df.total_value * 100)
    ax3.set_xlabel('Date')
    ax3.grid(True)


    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3, bymonthday=1))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))



    plt.tight_layout()

    return fig

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

    # Draw buying and selling points
    is_in_portfolio = (trade_df[ticker].values != 0).astype(int)
    is_in_portfolio = np.hstack([is_in_portfolio, [0]])


    buy_points = np.where((is_in_portfolio[:-1] == 0) & (is_in_portfolio[1:] == 1))[0]
    sell_points = np.where((is_in_portfolio[:-1] == 1) & (is_in_portfolio[1:] == 0))[0]

    for buy_point, sell_point in zip(buy_points,sell_points):
        buy_Date = trade_df.Date.values[buy_point]
        sell_Date = trade_df.Date.values[sell_point]

        ticker_buy_price = stocks_df[stocks_df.Date.values == buy_Date].Close.values[0]
        ticker_sell_price = stocks_df[stocks_df.Date.values == sell_Date].Close.values[0]

        ax1.plot(buy_Date, ticker_buy_price,marker='o', markersize=6, color='green', linewidth=2)
        ax1.plot(sell_Date, ticker_sell_price, marker='x',markersize=6, color='red',linewidth=2)

        profit = np.round((ticker_sell_price -ticker_buy_price) / ticker_buy_price * 100,1)
        #draw profit in middle of arc connecting buy and sell points
        # Get midpoint coordinates
        mid_date =  trade_df.Date.values[(buy_point+sell_point)//2]
        plot_price = (ticker_sell_price + ticker_buy_price) / 2
        plot_price = np.maximum(ticker_sell_price , ticker_buy_price)

        # Add some vertical offset for better visibility
        price_range = ticker_sell_price - ticker_buy_price
        offset = abs(price_range) * 0.1  # 10% offset
        plot_price = plot_price + offset

        ax1.annotate(f'{profit}%',
                     xy=(mid_date, plot_price),
                     ha='center', va='bottom',
                     fontsize=8, fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))

    # Plot 2: Portfolio Percentage
    if ticker in trade_df.keys():
        portfolio_percentages = trade_df[ticker] /trade_df.total_value * 100
    else:
        portfolio_percentages = trade_df.total_value*0



    ax2.fill_between(trade_df.Date,portfolio_percentages, alpha=0.6, color='lightblue', label='Portfolio %')
    ax2.plot(trade_df.Date,portfolio_percentages, 'darkblue', linewidth=1)
    ax2.set_ylabel('Portfolio %', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')
    ax2.grid(True)

    # Plot 2: RSI ...
    ax2_2 = ax2.twinx()
    ax2_2.plot(stocks_df.Date, stocks_df.rsi_14, 'y', linewidth=2, label='RSI')
    ax2_2.plot(stocks_df.Date, stocks_df.ma_rsi_14, 'c', linewidth=2, label='MA_RSI')
    # Set y-axis limits to fill the graph
    ax2.set_ylim([0, portfolio_percentages.max() * 1.05])
    ax2_2.set_ylim([0, 100])
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')



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
    #plt.show()
    return fig


def holding_per_time(trade_df , tickers_that_were_in_portfolio ):

    """Simple text-based visualization"""


    # prepare holding dict
    quarters = pd.date_range(start=trade_df.Date.min(), end=trade_df.Date.max(), freq='QS')
    holdings_dict = {}
    for quarter in quarters:
        quarter_df = trade_df[(trade_df.Date >= quarter) &  (trade_df.Date < quarter + pd.DateOffset(months=3)) ]
        holdings_dict[quarter.strftime('%Y-%m-%d')] = [ticker for ticker in tickers_that_were_in_portfolio if sum(quarter_df[ticker]) > 0]

    dates = list(holdings_dict.keys())

    max_stocks = np.max([len(v) for v in holdings_dict.values()])
    fig, ax = plt.subplots(figsize=(15, (max_stocks+6) // 3))

    for i, (date, stocks) in enumerate(holdings_dict.items()):
        # Plot vertical line for each date
        ax.axvline(x=i, color='lightgray', linestyle='--', alpha=0.5)

        # Add stock names as text
        stock_text = ', '.join(stocks)
        ax.text(i, 0.01, stock_text, rotation=90, ha='center', va='bottom',
                fontsize=7, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))

    ax.set_xlim(-0.5, len(dates) - 0.5)
    ax.set_xticks(range(len(dates)))

    ax.set_xticklabels(dates, rotation=45)
    #ax.set_ylim(0, 1)
    ax.set_ylabel('Holdings')
    ax.set_xlabel('Dates')
    ax.set_title('Stock Holdings Over Time')
    ax.grid(True, alpha=0.3)

    # Remove y-axis ticks as they're not meaningful
    ax.set_yticks([])

    plt.tight_layout()
    #plt.show()
    return fig




