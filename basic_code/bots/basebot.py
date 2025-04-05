import pandas as pd
import numpy as np
from typing import Dict
class tradeOrder:
    def __init__(self, order_type : str , amount : float = np.inf  ):
        self.order_type = order_type
        self.amount = amount


class BaseBot(object):
    def __init__(self, name: str = 'base' , params : Dict = None):
        self._name = name
        self._params = params

    def strategy(self, data: pd.DataFrame):
        '''

        :param data:
        :return:
        '''
        return np.full(len(data), tradeOrder('sell'))
    def display(self, stock_name : str, stock_df: pd.DataFrame ,
                trade_signal : np.array, reference_index : pd.DataFrame = None)->"fig":
        return None


