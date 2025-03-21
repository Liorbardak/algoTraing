import pylab as plt
import pandas as pd
import numpy as np
import os


def display_stock_price(datadir : str):
    '''
    basic view
    :param filepath:
    :return:
    '''
    filename = os.path.join(datadir, 'stockPrice.xlsx')
    df = pd.read_excel(filename, engine='openpyxl')
    df.keys()
    plt.plot(df['1. open'])
    plt.show()









if __name__ == "__main__":
    filepath = "C:\work\Algobot\data\INCY\IMVT"
    filepath = "C:\work\Algobot\data\INCY\IMXI"
    display_stock_price(filepath)