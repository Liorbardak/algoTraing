import pandas as pd
import numpy as np
class BaseBot(object):
    def __init__(self, name: str = 'base'):
        self._name = name

    def strategy(self, data: pd.DataFrame):
        '''

        :param data:
        :return:
        '''
        return np.full(len(data), 'sell')


