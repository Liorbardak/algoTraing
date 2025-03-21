import pylab as plt
import pandas as pd
import numpy as np
import os


def prerprocess_data(datadir : str )->pd.DataFrame:
    '''
    Create a common data frame  , with same dates for all stocks
    Add some features
    :param datadir: input directory
    :param mas: list of moving averages to calculate
    :param keyToUse: key to use for the ma
    :return: Dataframe with of all data
    '''
    # Minimal number of dates in a stock file
    minLengthtoUse = 300

    dirnames = [d for d in os.listdir(datadir) if os.path.isdir(os.path.join(datadir, d))]
    dfs = []
    for dirname in dirnames:
        filename = os.path.join(datadir, dirname, 'stockPrice.xlsx')
        df = pd.read_excel(filename, engine='openpyxl')
        print(f'{dirname} from {np.min(df.Date)}  to  {np.max(df.Date)}  {len(df)}')
        if(len(df)  < minLengthtoUse):
            continue
        # Add some features
        df['name'] = dirname

        dfs.append(df)
    df_all = pd.concat(dfs)

    # Take dates that has all stcks information
    Nstocks = len(set(df_all.name))
    filtered_df = []

    for date, df in df_all.groupby('Date'):
        if(len(df) ==Nstocks):
            filtered_df.append(df)



    df_all = pd.concat(filtered_df)
    if (len(df_all.Date) < minLengthtoUse):
        print('Error : no enough dates')
        return None
    return df_all


def get_average_stock(dfi : pd.DataFrame)->pd.DataFrame:
    '''
    Barbaric normalization - for each stock the first closing price will be 100
    :return: Data frame with of all data
    '''

    # Normalize
    df = dfi.copy()
    refData = np.min(df.Date)
    stock_names = set(df.name)
    for stock_name in stock_names:
       # normalize so first closing price will be 100
       normFact = 100 / df[(df.name == stock_name) & (df.Date == refData)]['4. close'].values[0]
       for k in ['1. open', '2. high', '3. low', '4. close']:
            df.loc[df.name == stock_name, k] = df[df.name == stock_name][k] * normFact

    # average on all stocks per time
    res = []
    for date, df_date in df.groupby('Date'):
        r = {'Date': date, 'name': 'average'}
        for k in ['1. open', '2. high', '3. low', '4. close',  '5. volume']:
            r[k] = df_date[k].mean()
        res.append(r)
    return pd.DataFrame(res)








if __name__ == "__main__":
    datadir = "C:\work\Algobot\data\INCY"
    df_all = prerprocess_data(datadir)
    #df_all.to_csv(os.path.join(datadir, 'stocks.csv'))

    avdf = get_average_stock(df_all)


