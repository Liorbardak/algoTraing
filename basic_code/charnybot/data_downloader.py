import yfinance as yf
import pandas as pd
import os
import numpy as np
import pylab as plt
stockdir =  'C:/Users/dadab/projects/algotrading/data/yahoo_tickers'

start = '2019-01-02'
end = '2025-10-01'

stocks_to_run_on =  ['APPLE']


ticker = '^GSPC'
data = yf.download('^GSPC', start=start, end=end ,auto_adjust=False )
snp_df = pd.DataFrame()
snp_df['Date'] = data[('Close', ticker)].index
for k in ['High', 'Low','Open','Close', 'Volume']:
    snp_df[k] = data[(k, ticker)].values

bad_stocks = []

for ticker in  stocks_to_run_on:

    print(ticker)

    data = yf.download(ticker, start=start, end=end,auto_adjust=False)
    if len(data) == 0:
        print(f"{ticker} not found ")
        # Can not load from yahoo - convert from existing data
        df = pd.DataFrame()

        continue

    df = pd.DataFrame()
    df['Date'] = data[('Close', ticker)].index
    for k in ['High', 'Low', 'Open', 'Close', 'Volume']:
        df[k] = data[(k, ticker)].values
    try:
        df['AdjClose'] = data[('Adj Close', ticker) ].values
    except:
        df['AdjClose'] =  df['Close']

    # add snp price
    snp_close_price = np.full(len(df), np.nan)
    for i, date in  enumerate(df['Date'].values):
        try:
            snp_close_price[i] = snp_df[snp_df.Date == date].Close.values[0]
        except:
            print('bad date')
    df['snp_Close'] = snp_close_price
    os.makedirs(os.path.join(stockdir, ticker), exist_ok=True)
    df.to_csv(os.path.join(os.path.join(stockdir, ticker, 'stockPrice.csv')))



print('bad_stocks')
print(bad_stocks)
plt.show()