import numpy as np
from CoffeeMeasure import *
from sklearn.neural_network import MLPRegressor


def feature_creation(data_frame):
    hsv_df = rgb2hsv(data_frame,
                     field_names=['rr', 'rg', 'rb'],
                     output_field_names=['h', 's', 'v'])

    return pd.concat([data_frame.reset_index(drop=True),
                      hsv_df.reset_index(drop=True)], axis=1)


class MLPModel(CoffeeMeasureCore):
    essentials = ['rr', 'rb', 'rg', 'rc']
    model_name = "MLP"

    def __dump_model__(self):
        if self.model is None:
            self.log.error("Model not available, have you train it?")
            return {}
        return {'model_name': self.model_name,
                'reg_coef': self.numpy_array_flattener(self.model.coefs_),
                'reg_intercept': self.numpy_array_flattener(self.model.intercepts_),
                'note': 'y = ( Σ((X * mlp_coef[i]) + mlp_intercept[i]))*128; where r,g,b/=127, rc/=511'}

    def __evaluate__(self, eval_data_frame):
        if self.model is None:
            self.log.error('Please train the model before you proceeding.')
            return {}
        #
        self.check_essential_params(eval_data_frame)
        self.data_pre_processing(eval_data_frame)
        eval_data_frame = feature_creation(eval_data_frame)
        #
        X, y = eval_data_frame[['h', 's', 'v', 'rc']].to_numpy(), \
               eval_data_frame['value'].to_numpy() / 127
        #
        pred = self.model.predict(X)
        self.report(groundTruth=y, predicted=pred)

    def __logic__(self, hyper_params):
        self.data_frame = feature_creation(self.data_frame)
        #
        X, y = self.data_frame[['h', 's', 'v', 'rc']].to_numpy(), \
               self.data_frame['value'].to_numpy() / 127

        default_parameters = {
            'hidden_layer_sizes': (100, 50, 25), 'max_iter': 50000,
            'solver': 'sgd', 'random_state': 5566, 'early_stopping': True,
            'activation': 'tanh', 'learning_rate': 'invscaling'
        }
        default_parameters.update(hyper_params)

        self.model = MLPRegressor(**default_parameters)
        self.model.fit(X, y)
        #

    def __init__(self, data_frame, log_level=30):
        self.data_frame = data_frame
        super().__init__(log_level)
        self.log.info("Essentials : {}".format(self.essentials))
