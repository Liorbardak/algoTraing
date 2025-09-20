import yfinance as yf
import pandas as pd
import os
import numpy as np
from datetime import datetime, timedelta
import pylab as plt

def get_historical_iv_from_numpy_date(symbol, numpy_date):
    """
    Get historical IV using numpy.datetime64 input
    """
    # Convert numpy.datetime64 to Python datetime
    target_date = pd.Timestamp(numpy_date).to_pydatetime()

    #print(f"Converted numpy date to: {target_date}")

    ticker = yf.Ticker(symbol)

    try:
        # Get current options data (limitation of yfinance)
        exp_dates = ticker.options

        if not exp_dates:
            print(f"No options data available for {symbol}")
            return None

        options = ticker.option_chain(exp_dates[0])
        calls = options.calls

        # print(f"⚠️  WARNING: yfinance typically only has current options data")
        # print(f"This may not reflect IV from {target_date.date()}")

        # Get historical stock price for the target date
        start_date = target_date
        end_date = target_date + timedelta(days=5)  # Get a few days to ensure we get data

        hist_price = ticker.history(start=start_date, end=end_date)

        if hist_price.empty:
            print(f"No price data available for {target_date.date()}")
            return None

        price_on_date = hist_price['Close'].iloc[0]
        actual_date = hist_price.index[0]

        # Find ATM options based on historical price
        calls['strike_diff'] = abs(calls['strike'] - price_on_date)
        atm_call = calls.loc[calls['strike_diff'].idxmin()]

        result = {
            'requested_date': target_date.date(),
            'actual_price_date': actual_date.date(),
            'stock_price': price_on_date,
            'atm_strike': atm_call['strike'],
            'current_implied_volatility': atm_call['impliedVolatility'],
            'option_expiration': exp_dates[0],
            'note': 'This shows current IV, not historical IV from the requested date'
        }

        return result['current_implied_volatility']

    except Exception as e:
        print(f"Error: {e}")
        return None



# Prevent sleep in windows
import ctypes

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


stockdir1 =  'C:/Users/dadab/projects/algotrading/data/yahoo_tickers'
data_info_dir =  'C:/Users/dadab/projects/algotrading/data/tickers'

min_start = '2019-01-02'
max_end = '2025-10-01'

stocks_to_run_on = [entry.name for entry in os.scandir(stockdir1) if entry.is_dir()]
vix = yf.download("^VIX", start=min_start, end=max_end)
vix_date = vix[('Close', "^VIX")].index
vix_close = vix[('Close', "^VIX")].values
for ticker in  stocks_to_run_on:
    print(ticker)
    df = pd.read_csv(f'{stockdir1}/{ticker}/StockPrice.csv')
    # Add vix
    df['vix_Close'] = np.nan
    for ii in range(len(df.Date.values)):
        ind = np.where(vix_date == df.Date.values[ii])
        if (len(ind[0])):
            df.loc[ii, 'vix_Close'] = vix_close[ind[0][0]]

   #market cap
    df['MarketCap'] = np.nan
    if os.path.isfile(f'{data_info_dir}/{ticker}/financial_data.xlsx'):
        histval = pd.read_excel(f'{data_info_dir}/{ticker}/financial_data.xlsx')

        if(len(histval)):
            dates  = [np.datetime64(d) for d in df['Date']]
            for ii in range(len(histval.reportedDate.values)):
                ind = np.where(dates == histval.reportedDate.values[ii])
                if(len(ind[0])):
                    df.loc[ind[0][0], 'MarketCap'] = histval.MarketCap.values[ii]
            df.to_csv(f'{stockdir1}/{ticker}/StockPrice.csv')
        else:
            print(f'No data {ticker}')
    # iv
    df['iv'] = np.nan
    for date in df.Date.values:
        iv = get_historical_iv_from_numpy_date(ticker, np.datetime64(date))
        df.loc[df.Date == date, 'iv'] = iv




