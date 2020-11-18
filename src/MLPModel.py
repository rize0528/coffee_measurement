import numpy as np
from CoffeeMeasure import *
from sklearn.neural_network import MLPRegressor


def feature_creation(data_frame):
    hsv_df = rgb2hsv(data_frame,
                     field_names=['norm_rr', 'norm_rg', 'norm_rb'],
                     output_field_names=['h', 's', 'v'])

    return pd.concat([data_frame.reset_index(drop=True),
                      hsv_df.reset_index(drop=True)], axis=1)


class Model(CoffeeMeasureCore):
    essentials = ['rr', 'rb', 'rg', 'rc']
    model_name = "MLP"

    def __dump_model__(self):
        if self.model is None:
            self.log.error("Model not available, have you train it?")
            return {}
        return {
                    'model_name': self.model_name,
                    'mlp_weights': self.numpy_array_flattener(self.model.coefs_),
                    'mlp_bias': self.numpy_array_flattener(self.model.intercepts_),
                    'reconstruct': 'y(i+1) = tanh(dot(y(i), mlp_weights(i)) + mlp_bias(i), where i is in {1 to L-1} '
                                   ', y0 is X and L is the number of layers of MLP model (including input and output '
                                   'layers). y(L) = dot(y(L-1), mlp_weight(L-1)) + mlp_bias(L-1), y(L) is the MLP '
                                   'prediction value.',
                    'note': 'Since the memory size of arduino are quite small, too much layers or neurons would lead '
                            'arduino popup failures during run-time. So we suggest the total number of parameter should'
                            ' lower than 250.'
               }

    def __evaluate__(self, eval_data_frame):
        if self.model is None:
            self.log.error('Please train the model before you proceeding.')
            return {}
        #
        self.check_essential_params(eval_data_frame)
        self.data_pre_processing(eval_data_frame)
        eval_data_frame = feature_creation(eval_data_frame)
        #
        X, y = eval_data_frame[['h', 's', 'v']].to_numpy(), \
               eval_data_frame['value'].to_numpy() / 127
        #
        pred = self.model.predict(X)
        self.report(groundTruth=y, predicted=pred, ascii_chart_config={'display_width': 80})

    def __logic__(self, hyper_params):
        self.data_frame = feature_creation(self.data_frame)
        #
        X, y = self.data_frame[['h', 's', 'v']].to_numpy(), \
               self.data_frame['value'].to_numpy() / 127

        default_parameters = {
            'hidden_layer_sizes': (6, 5, 5), 'max_iter': 50000,
            'solver': 'adam', 'random_state': 9527, 'early_stopping': True,
            'activation': 'tanh', 'learning_rate': 'adaptive'
        }
        default_parameters.update(hyper_params)

        self.model = MLPRegressor(**default_parameters)
        self.model.fit(X, y)
        #

    def __init__(self, data_frame, log_level=30):
        self.data_frame = data_frame
        super().__init__(log_level)
        self.log.info("Essentials : {}".format(self.essentials))
