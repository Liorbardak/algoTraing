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
    #ax1.set_ylim(80, 230)

    # Plot 2: Portfolio Percentage
    if ticker in trade_df.keys():
        portfolio_percentages = trade_df[ticker] /trade_df.total_value
    else:
        portfolio_percentages = trade_df.total_value*0

    ax2.fill_between(trade_df.Date,portfolio_percentages, alpha=0.6, color='lightblue', label='Portfolio %')
    ax2.plot(trade_df.Date,portfolio_percentages, 'darkblue', linewidth=1)
    ax2.set_ylabel('Portfolio %', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, max(portfolio_percentages) * 1.1+0.1)
    ax2.legend(loc='upper right')
    ax2.grid(True)

    # Create stacked bar chart
    width = 15  # Width of bars in days
    analyst_dates =complement_df.Date
    ax3.bar(analyst_dates, complement_df.number_of_analysts_with_compliments, width,
            color='lightcoral', alpha=0.7, label='Number analysts with compliments')
    #ax3.bar(analyst_dates, complement_df.number_of_analysts_comp_1, width,
    #        color='lightcoral', alpha=0.7, label='Level 1')
    #ax3.bar(analyst_dates,complement_df.number_of_analysts_comp_2, width,
    #        bottom=complement_df.number_of_analysts_comp_1,
    #        color='lightblue', alpha=0.7, label='Level 2')
    #ax3.bar(analyst_dates, complement_df.number_of_analysts_comp_3, width,
    #        bottom=complement_df.number_of_analysts_comp_2,
    #        color='lightgreen', alpha=0.7, label='Level 3')

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

    #ax3.xaxis.set_major_locator(mdates.YearLocator())
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




