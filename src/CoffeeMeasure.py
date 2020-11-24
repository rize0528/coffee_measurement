import os
import sys
import json
import logging
import importlib
import numpy as np
import pandas as pd
import pickle as pkl
import asciichartpy as chart
from utils import *
from typing import Union
from sklearn.metrics import mean_squared_error, mean_absolute_error

WD = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(WD, '../gy33-agtron'))
param_gen = __import__("gen-param-h")
#param_gen = importlib.import_module("param_generator", "gen-param-h")

def feature_creation(data_frame):
    hsv_df = rgb2hsv(data_frame,
                     field_names=['norm_rr', 'norm_rg', 'norm_rb'],
                     output_field_names=['h', 's', 'v'])

    return pd.concat([data_frame.reset_index(drop=True),
                      hsv_df.reset_index(drop=True)], axis=1)


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

        # Normalize raw rgb with rc.
        for col in columns:
            if col in ['rr', 'rg', 'rb']:
                _df['norm_{}'.format(col)] = _df[col] / _df['rc']

        # Normalize the rest channels.
        for col in columns:
            if col == 'ct':
                _df[col] /= 8191
            elif col == 'lux':
                _df[col] /= 31
            elif col == 'rc':
                _df['raw_rc'] = _df[col].copy()
                _df[col] /= 511
            elif col in self.colour_channels:
                _df[col] /= 255

    def report_string(self, groundTruth: Union[np.ndarray, pd.core.frame.DataFrame],
               predicted: Union[np.ndarray, pd.core.frame.DataFrame],
               ascii_chart_config: dict = {}):
        _pred = predicted.copy() * 128
        _gt = groundTruth.copy() * 128
        error = (predicted - groundTruth) * 128
        #
        report_string = '''
        Evaluation report:
           |- Error values = Max:{0:.4f}, Min:{1:.4f}
           |- Median error value = {2:.4f}
           |- Standard deviation = {3:.4f}
           |- Loss score (MSE) = {4:.4f}
           |-            (MAE) = {5:.4f}
        '''.format(np.max(error),
                   np.min(error),
                   np.median(error),
                   np.std(error),
                   mean_squared_error(_gt, _pred),
                   mean_absolute_error(_gt, _pred))
        config = {
            'height': 15,
            'display_width': 100
        }
        _display = config['display_width']

        out_chart = chart.plot(series=[(_gt - _pred).tolist()[:_display]], cfg=config)
        table_width = np.max(list(map(lambda x: len(x), out_chart.split('\n'))))

        report_string = report_string + "\n" + "="*table_width + "\n" + out_chart + "\n" + "="*table_width

        return report_string

    def report(self, groundTruth: Union[np.ndarray, pd.core.frame.DataFrame],
               predicted: Union[np.ndarray, pd.core.frame.DataFrame],
               ascii_chart_config: dict = {}):
        _pred = predicted.copy() * 128
        _gt = groundTruth.copy() * 128
        error = (predicted - groundTruth) * 128

        self.log.info('Evaluation report:')
        self.log.info(' |- Max error: {}'.format(np.max(error)))
        self.log.info(' |- Min error: {}'.format(np.min(error)))
        self.log.info(' |- Average error: {}'.format(np.mean(error)))
        self.log.info(' |- Average absolute error: {}'.format(np.mean(np.abs(error))))
        self.log.debug(' |- Loss value between ground truth and prediction: {}'.format(error))
        #
        config = {
            'colors': [
                chart.red,
                chart.green,
                chart.blue
            ],
            'height': 15, 'max': 128,
            'display_width': 100
        }
        config.update(ascii_chart_config)
        #
        legend = "+{:=^20}+\n|{:^20}|\n|{:^20}|\n|{:^20}|\n+{:=^20}+" \
            .format('Legend', 'Green: Prediction', 'Red: Ground Truth', 'Blue: Error value', '')
        #
        _display = config['display_width']
        #
        out_chart = chart.plot(series=[_gt.tolist()[:_display], _pred.tolist()[:_display],
                                       (_pred - _gt).tolist()[:_display]], cfg=config)
        log_msg = "\nKernel type: {:=^20}".format(self.model_name) + "\n" + \
                  "\n" + legend + "\n" + "+===[Performance report on evaluation data]===+" + "\n" + out_chart
        self.log.info(log_msg)

    def train(self, eval_df: pd.DataFrame = None,  hyper_params: dict = {}, apply_evaluation = True):
        if eval_df is None:
            self.log.info("No available evaluation data could adopt, use training data instead.")
            eval_df = self.raw_data_frame.copy()
        self.log.debug("Given hyper-parameters: {}".format(json.dumps(hyper_params)))
        self.__logic__(hyper_params)
        if apply_evaluation:
            self.__evaluate__(eval_df)

    def predict(self, pred_df: pd.DataFrame):
        op_df = pred_df.copy()
        self.check_essential_params(op_df)
        self.data_pre_processing(op_df)
        op_df = feature_creation(op_df)
        X, y = op_df[['h', 's', 'v']].to_numpy(), \
               op_df['value'].to_numpy() / 127
        Xp = self.__wrap__(X)
        return self.model.predict(Xp), y

    def numpy_array_flattener(self, array: Union[list, np.ndarray]):
        if isinstance(array, np.ndarray):
            return array.tolist()
        return [self.numpy_array_flattener(x) for x in array]

    def dump_params(self, path: str):
        with open(path, 'w') as wp:
            model_dict = self.__dump_model__()
            param_gen.param_generator(model_dict, wp)
            self.log.info('Arduino config file has been saved to: {}'.format(path))

    def dump_model(self, path: str):
        with open(path, 'w') as wp:
            model_str = self.__dump_model__()
            wp.write(json.dumps(model_str))
            self.log.info('Model saved to: {}'.format(path))

    def __init__(self, log_level: int = logging.INFO):
        #
        self.raw_data_frame = self.data_frame.copy()
        self.base_essentials = ['value']
        log_format = '[%(asctime)-15s][%(levelname)s][%(filename)s] %(message)s'
        logging.basicConfig(level=log_level, format=log_format)
        self.log = logging.getLogger()
        #
        self.check_essential_params()
        self.data_pre_processing()
