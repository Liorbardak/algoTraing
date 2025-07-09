import os
import numpy as np
import json
import pandas as pd
import re
import shutil
def compliment_format_converter(inpath , outpth):
    os.makedirs(outpth, exist_ok=True)
    tickers = [re.match(r'^([A-Z]+)', file).group(1) for file in os.listdir(inpath)]

    for ticker in os.listdir(inpath):
        if os.path.isdir(os.path.join(inpath, ticker)) == False:
            continue
        summery_compfile = os.path.join(inpath, ticker, 'summarizeCompliments_gpt41_2023.json')
        if os.path.isfile(summery_compfile):
            shutil.copy(summery_compfile, os.path.join(outpth, f'{ticker}_compliment_summary.json'))


def calculate_average_ticker(tickers_df):

    # Calculate the average of all stocks

    avg_df = None
    keys_to_avg = ['High', 'Low', 'Open', 'Close', 'AdjClose', 'Volume']

    # Calculate the average of all stocks
    for tdf in tickers_df:
        # Get the stock in the time range
        tdf = tdf[(pd.to_datetime(tdf.Date) >= actual_min_max_dates[0]) & (
                    pd.to_datetime(tdf.Date) <= actual_min_max_dates[1])]
        # Normalize price by the first date
        for k in keys_to_avg:
            tdf[k] = tdf[k] / tdf.Close.values[0]
        if avg_df is None:
            avg_df = tdf
        else:
            for k in keys_to_avg:
                avg_df[k] = avg_df[k].values + tdf[k].values

    for k in keys_to_avg:
        avg_df[k] = avg_df[k].values / len(tickers_df)



if __name__ == "__main__":
    inpath = 'C:/Users/dadab/projects/algotrading/data/gpt41_pretest'
    outpth = 'C:/Users/dadab/projects/algotrading/data/complements/gpt41_2023'

    compliment_format_converter(inpath, outpth)
