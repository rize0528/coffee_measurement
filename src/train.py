import os, sys
import logging
import argparse
import pandas as pd
from itertools import chain
from RegressionModel import *

SCRIPT_WD = os.path.dirname(os.path.realpath(__file__))
DEFAULT_TRAINING_DATA = os.path.join(SCRIPT_WD, "../res/default_training.csv")
SUPPORTED_MODELS = {
    'regression': ['regression', 'reg'],
    'function_approximation': ['function_approximation', 'fa'],
    'auto_encoder': ['auto_encoder', 'ae']
}

logging.basicConfig(level=logging.DEBUG, format=
                    '[%(asctime)-15s][%(levelname)s][%(filename)s] %(message)s')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', default=DEFAULT_TRAINING_DATA, type=str)
    parser.add_argument('-m', '--model',
                        choices=list(chain(*SUPPORTED_MODELS.values())),
                        required=True)
    args = parser.parse_args()
    #
    loaded_df = []
    if os.path.isdir(args.input):
        # Load all csv files
        _files = list(map(lambda y: os.path.join(args.input, y),
                          filter(lambda x: x.endswith('.csv'), os.listdir(args.input))))
        _loaded, _failed = 0, 0
        for _file in _files:
            try:
                loaded_df.append(pd.read_csv(_file))
                _loaded += 1
            except Exception as e:
                logging.warning('"{0}" can\'t be loaded.'.format(_file))
                logging.warning(str(e))
                _failed += 1
        logging.info('{0} file(s) were loaded.'.format(_loaded))
        if _failed > 0:
            logging.info('{0} file(s) were failed to load.'.format(_failed))
    elif args.input.lower().endswith('.csv'):
        # Load csv file
        loaded_df.append(pd.read_csv(args.input))
        logging.info('"{0}" loaded'.format(args.input.lower()))
    else:
        logging.warning("Invalid file type. aborted")
        return 1
    #
    df = pd.concat(loaded_df, axis=0)
    #
    if args.model in SUPPORTED_MODELS['regression']:
        cmf = RegressionModel(df)
        cmf.train()
        cmf.dump_model_pickle('../output/reg_demo.pkl')
    elif args.model in SUPPORTED_MODELS['function_approximation']:
        pass
    else:
        raise NotImplementedError('Model "{}" are currently not supported.'.format(args.model))
    return 0

if __name__ == "__main__":
    sys.exit(main())
