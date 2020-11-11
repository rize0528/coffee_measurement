import json
import logging
import numpy as np
import pandas as pd
import pickle as pkl
import asciichartpy as chart
from utils import *
from typing import Union


class CoffeeMeasureCore:
    """
        Required variables:
            essentials:  an array contains field names that the model would use.
            data_frame:  Pandas dataframe created in train.py
    """
    colour_channels = ['rr', 'rg', 'rb', 'cr', 'cg', 'cb']
    model = None
    model_name = 'core'

    def check_essential_params(self, external_df=None):
        _essentials = self.base_essentials + self.essentials
        _df = external_df if external_df is not None else self.data_frame
        fields_not_found = [x for x in _essentials if x not in _df.columns]
        #
        if len(fields_not_found) > 0:
            self.log.ERROR('Lacks of essential parameters for {0}'.format(self.__name__))

    def data_pre_processing(self, external_df=None):
        _df = external_df if external_df is not None else self.data_frame
        columns = _df.columns

        for col in columns:
            if col == 'ct':
                _df[col] /= 8191
            elif col == 'lux':
                _df[col] /= 31
            elif col == 'rc':
                _df[col] /= 511
            elif col in self.colour_channels:
                _df[col] /= 255

    def report(self, groundTruth: Union[np.ndarray, pd.core.frame.DataFrame],
               predicted: Union[np.ndarray, pd.core.frame.DataFrame]):
        _pred = predicted.copy() * 128
        _gt = groundTruth.copy() * 128
        error = (predicted - groundTruth) * 128

        self.log.info('Linear regression evaluation report:')
        self.log.info(' |- Max error: {}'.format(np.max(error)))
        self.log.info(' |- Min error: {}'.format(np.min(error)))
        self.log.info(' |- Average error: {}'.format(np.mean(error)))
        self.log.debug(' |- Loss value between ground truth and prediction: {}'.format(error))

        config = {
            'colors': [
                chart.red,
                chart.green,
                chart.blue
            ],
            'height': 20, 'max': 128
        }
        #
        legend = "+{:=^20}+\n|{:^20}|\n|{:^20}|\n|{:^20}|\n+{:=^20}+" \
            .format('Legend', 'Green: Prediction', 'Red: Ground Truth', 'Blue: Error value', '')
        out_chart = chart.plot(series=[_gt.tolist(), _pred.tolist(),
                                       (_pred - _gt).tolist()], cfg=config)
        self.log.info("\n" + legend)
        self.log.info("\nKernel type: {:=^20}".format(self.model_name))
        self.log.info("\n" + out_chart)

    def train(self, hyper_params: dict):
        _eval_df = self.raw_data_frame.copy()
        self.__logic__(hyper_params)
        self.__evaluate__(_eval_df)

    def numpy_array_flattener(self, array: Union[list, np.ndarray]):
        if isinstance(array, np.ndarray):
            return array.tolist()
        return [self.numpy_array_flattener(x) for x in array]

    def dump_model(self, path: str):
        with open(path, 'w') as wp:
            model_str = self.__dump_model__()
            wp.write(json.dumps(model_str))
            self.log.info('Model saved to: {}'.format(path))

    def __init__(self, log_level: int = logging.INFO):
        #
        self.raw_data_frame = self.data_frame.copy()
        self.base_essentials = ['value', 'label']
        log_format = '[%(asctime)-15s][%(levelname)s][%(filename)s] %(message)s'
        logging.basicConfig(level=log_level, format=log_format)
        self.log = logging.getLogger()
        #
        self.check_essential_params()
        self.data_pre_processing()
