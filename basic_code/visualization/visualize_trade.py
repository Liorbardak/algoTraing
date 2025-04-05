from typing import Dict

import pandas as pd


from basic_code.utils.report_utils import HtmlReport


def visualize_trade(trade_info : Dict , stocks_df : pd.DataFrame ,  reference_index : pd.DataFrame , report : "HtmlReport" = None):
    '''
    Visualize bot results
    :param trade_info:
    :param stocks_df:
    :param reference_index:
    :param report:
    :return:
    '''
    import matplotlib
    matplotlib.use('Qt5Agg')
    import pylab as plt
    fig = plt.figure()
    plt.plot(trade_info['total_value'] / trade_info['total_value'][0]*100 , label = 'trade')
    plt.plot(reference_index['price'], label = 'reference index ')
    plt.title('overall balance')
    plt.legend()
    plt.ylabel('price')
    if report:
        report.add_figure('overall balance' , fig)
    plt.close("all")
    return

    for si, (stock_name, stock_df) in enumerate(stocks_df.groupby('name', sort=True)):
        stock_df = stock_df.reset_index()

        fig, ax1 = plt.subplots()

        ax2 = ax1.twinx()
        ax1.plot(trade_info['stocks_per_share'][si], 'g-')
        ax2.plot(stock_df.price / stock_df.price.values[0], 'b-')
        ax1.set_ylabel('number_of_stocks', color='g')
        ax2.set_ylabel('price', color='b')

        plt.title(f' {stock_name}')

        if report:
            report.add_figure(stock_name, fig)
