import pylab as plt
import pandas as pd
import numpy as np
import os


def prerprocess_data(datadir : str , minLengthtoUse :int = 300):
    '''
    Prepare data for work -
    - Filter out stocks that does not have enough information
    - Take only dates that has data from all stocks
    - Create reference index - average of all stocks
    - Save the reference index and the common stocks data frame
    :param datadir: input directory
    :param minLengthtoUse:  Minimal number of dates in a stock file directory

    '''


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

    # Take dates that has all stocks information
    Nstocks = len(set(df_all.name))
    filtered_df = []

    for date, df in df_all.groupby('Date'):
        if(len(df) ==Nstocks):
            filtered_df.append(df)



    df_all = pd.concat(filtered_df).reset_index()

    if (len(set(df_all.Date)) < minLengthtoUse):
        print('Error : no enough dates')
        return

    print(f' preroc {datadir} #stocks {len(set(df_all.name))} #dates {len(set(df_all.Date))}  from {np.min(df_all.Date)} to {np.max(df_all.Date)} ')
    # Save data
    df_all.to_csv(os.path.join(datadir, 'all_stocks.csv'))
    # Get & Save  average index
    avgdata = get_average_stock(df_all)
    avgdata.to_csv(os.path.join(datadir, 'reference_index.csv'))




def get_average_stock(dfi : pd.DataFrame)->pd.DataFrame:
    '''
    Average all stocks with equal weights
    Normalization - for each stock, set the first closing price will be 100
    :return: average dataframe
    '''
    reference_key = '4. close'
    keys_to_average =  ['1. open', '2. high', '3. low', '4. close']

    # Normalize
    df = dfi.copy()
    refData = np.min(df.Date)
    stock_names = set(df.name)
    for stock_name in stock_names:
       # normalize so first closing price will be 100
       normFact = 100 / df[(df.name == stock_name) & (df.Date == refData)][reference_key].values[0]
       for k in keys_to_average:
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
    dataindir = "C:\work\Algobot\data\INCY"
    prerprocess_data(dataindir)

    #df_all.to_csv(os.path.join(datadir, 'stocks.csv'))



