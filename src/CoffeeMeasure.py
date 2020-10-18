import logging
import pickle as pkl
from utils import *

class CoffeeMeasureCore:
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

    def train(self, **hyper_params):
        _eval_df = self.raw_data_frame.copy()
        self.__logic__()
        self.__evaluate__(_eval_df)

    def dump_model_pickle(self, path):
        with open(path, 'wb') as wp:
            pkl.dump(self.__dump_model__(), wp)
            self.log.info('Model saved to: {}'.format(path))

    def __init__(self, log_level=logging.INFO):
        #
        self.raw_data_frame = self.data_frame.copy()
        self.base_essentials = ['value', 'label']
        log_format = '[%(asctime)-15s][%(levelname)s][%(filename)s] %(message)s'
        logging.basicConfig(level=log_level, format=log_format)
        self.log = logging.getLogger()
        #
        self.check_essential_params()
        self.data_pre_processing()
