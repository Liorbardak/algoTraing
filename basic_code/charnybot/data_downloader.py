import yfinance as yf
import pandas as pd
import os
import numpy as np
import pylab as plt
stockdir =  'C:/Users/dadab/projects/algotrading/data/tickers'
# outdir =  'C:/Users/dadab/projects/algotrading/data/snp500_yahoo'
# bad = []
# bad_length = []
# snp = pd.read_csv(os.path.join(stockdir, 'sp500_stocks.csv'))

start = '2019-01-02'
end = '2025-06-01'

stocks_to_run_on =  ['ARDX', 'BVS']


ticker = '^GSPC'
data = yf.download('^GSPC', start=start, end=end ,auto_adjust=False )
snp_df = pd.DataFrame()
snp_df['Date'] = data[('Close', ticker)].index
for k in ['High', 'Low','Open','Close', 'Volume']:
    snp_df[k] = data[(k, ticker)].values

bad_stocks = []

for ticker in  stocks_to_run_on:

    print(ticker)

    try:
        stock_price = pd.read_parquet(os.path.join(stockdir, ticker, 'stockPrice_corrected.parquet'))
    except:
        stock_price = pd.read_parquet(os.path.join(stockdir, ticker, 'stockPrice.parquet'))


    data = yf.download(ticker, start=start, end=end,auto_adjust=False)
    if len(data) == 0:
        print(f"{ticker} not found ")
        # Can not load from yahoo - convert from existing data
        df = pd.DataFrame()
        # df['Date'] = stock_price['Date']
        # df['High'] = stock_price['close']
        # df['Low'] = stock_price['close']
        # df['Open'] = stock_price['close']
        # df['Close'] = stock_price['close']
        # df['Volume'] = 0
        # df['AdjClose'] = stock_price['close']
        # df['snp_Close'] = snp_close_price
        # df.to_csv(os.path.join(os.path.join(stockdir, ticker, 'stockPrice.csv')))

        continue

    df = pd.DataFrame()
    df['Date'] = data[('Close', ticker)].index
    for k in ['High', 'Low', 'Open', 'Close', 'Volume']:
        df[k] = data[(k, ticker)].values
    try:
        df['AdjClose'] = data[('Adj Close', ticker) ].values
    except:
        df['AdjClose'] =  df['Close']



    mindate = np.min([stock_price.Date.min(), df.Date.min()])
    maxdate = np.min([stock_price.Date.max(), df.Date.max()])
    diff = (df[(df.Date >= mindate) & (df.Date <= maxdate)].Close -   stock_price[(stock_price.Date >= mindate) & (stock_price.Date <= maxdate)].close).values
    date_diff =  df[(df.Date >= mindate) & (df.Date <= maxdate)].Date
    maxdiff = np.max(np.abs(diff))

    print(f"{ticker} max diff {maxdiff}")
    if maxdiff > 1e-1:

        bad_stocks.append(ticker)
        #Display
        plt.figure()
        plt.plot(data.index, data.Close, label='yahoo close' , alpha=0.5)
        plt.plot(data.index, df['AdjClose'], label='yahoo adj close', alpha=0.5)

        plt.plot(stock_price.Date, stock_price.close, label='legacy close ', alpha=0.5)
        plt.plot(date_diff, diff ,label='diff ')
        plt.legend()
        plt.title(ticker)
       # plt.show()

    # add snp price
    snp_close_price = np.full(len(df), np.nan)
    for i, date in  enumerate(df['Date'].values):
        try:
            snp_close_price[i] = snp_df[snp_df.Date == date].Close.values[0]
        except:
            print('bad date')
    df['snp_Close'] = snp_close_price
    df.to_csv(os.path.join(os.path.join(stockdir, ticker, 'stockPrice.csv')))


    # Display
    # plt.plot(data.index, data.Close, label='yahoo close')
    #
    # #plt.plot(data.index, data.Open, label='yahoo open')
    # # plt.plot(stock_price.Date, stock_price.close, label='legacy close ')
    # # plt.plot(stock_price.Date, stock_price['1. open'], label='legacy open ')
    # plt.plot(stock_price.Date, stock_price.close, label='legacy close ')
    # #plt.plot(stock_price.Date, stock_price['1. open'], label='legacy open ')

    # plt.legend()
    # plt.title(ticker)
    # plt.show()

    #df.to_csv(os.path.join(outdir, ticker + '.csv'))

print('bad_stocks')
print(bad_stocks)
plt.show()