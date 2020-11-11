import os, sys
import json
import logging
import argparse
import pandas as pd
from itertools import chain
from datetime import datetime
from RegressionModel import *
from MLPModel import *

SCRIPT_WD = os.path.dirname(os.path.realpath(__file__))
DEFAULT_TRAINING_DATA = os.path.join(SCRIPT_WD, "../res/default_training.csv")
SUPPORTED_MODELS = {
    'regression': ['regression', 'reg'],
    'mlp_regressor': ['mlp_regressor', 'mlp'],
    'auto_encoder': ['auto_encoder', 'ae']
}

def main():
    """
        Supported Models:
            Linear regression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html?highlight=linear%20regression#sklearn.linear_model.LinearRegression
            MLP Regressor:  https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html?highlight=mlpregressor#sklearn.neural_network.MLPRegressor
        You can find available hyper-parameter in above links.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', default=DEFAULT_TRAINING_DATA, type=str)
    parser.add_argument('-m', '--model',
                        choices=list(chain(*SUPPORTED_MODELS.values())),
                        required=True)
    parser.add_argument('-o', '--output', default=os.path.join(SCRIPT_WD, '../output'))
    parser.add_argument('-l', '--log-level', choices=["debug", "info", "warning", "error"],
                        default="info", dest="log_level")
    parser.add_argument('-p', '--hyper-parameters', dest='hyper_parameters', default="{}", help="hyper parameters")
    args = parser.parse_args()
    #
    log_level = {"debug": 10, "info": 20, "warning": 30, "error": 40}.get(args.log_level)

    logging.basicConfig(level=log_level, format=
    '[%(asctime)-15s][%(levelname)s][%(filename)s] %(message)s')

    try:
        hyperp = json.loads(args.hyper_parameters)
    except Exception as e:
        logging.error("Unable to parse given hyper-parameter string. Program halt...")
        raise str(e)

    # Create output folder if not exist.
    os.makedirs(os.path.realpath(args.output), exist_ok=True)

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
    output_filepath = os.path.join(args.output,
                                   'model_{}.json'.format(datetime.now().strftime("%m%d_%H%M")))
    #
    if args.model in SUPPORTED_MODELS['regression']:
        rm = RegressionModel(df, {'log_level': log_level})
        rm.train(hyperp)
        rm.dump_model(output_filepath)
    elif args.model in SUPPORTED_MODELS['mlp_regressor']:
        mlp = MLPModel(df, {'log_level': log_level})
        mlp.train(hyperp)
        mlp.dump_model(output_filepath)
    else:
        raise NotImplementedError('Model "{}" are currently not supported.'.format(args.model))
    return 0


if __name__ == "__main__":
    sys.exit(main())
