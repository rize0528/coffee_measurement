import numpy as np
from CoffeeMeasure import *
from sklearn.linear_model import LinearRegression


def feature_creation(data_frame):
    hsv_df = rgb2hsv(data_frame,
                     field_names=['norm_rr', 'norm_rg', 'norm_rb'],
                     output_field_names=['h', 's', 'v'])

    return pd.concat([data_frame.reset_index(drop=True),
                      hsv_df.reset_index(drop=True)], axis=1)


class Model(CoffeeMeasureCore):
    essentials = ['rr', 'rb', 'rg', 'rc']
    model_name = "linear_regression"

    def __dump_model__(self):
        if self.model is None:
            self.log.error("Model not available, have you train it?")
            return {}
        return {'model_name': self.model_name,
                'reg_coef': self.model.coef_.tolist(),
                'reg_intercept': self.model.intercept_.tolist(),
                'reconstruct': 'y = (dot(X, reg_coef) + reg_intercept) * 128',
                'note': 'The linear regression model to describe a distribution with a linear function.'}

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
        self.report(groundTruth=y, predicted=pred)

    def __logic__(self, hyper_params):
        self.data_frame = feature_creation(self.data_frame)
        #
        X, y = self.data_frame[['h', 's', 'v']].to_numpy(), \
               self.data_frame['value'].to_numpy() / 127
        self.model = LinearRegression(**hyper_params)
        self.model.fit(X, y)
        #

    def __init__(self, data_frame, log_level=30):
        self.data_frame = data_frame
        super().__init__(log_level)
        self.log.info("Essentials : {}".format(self.essentials))
