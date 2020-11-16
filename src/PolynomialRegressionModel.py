import numpy as np
import statsmodels.api as sm
from CoffeeMeasure import *
from sklearn.preprocessing import PolynomialFeatures


def feature_creation(data_frame):
    hsv_df = rgb2hsv(data_frame,
                     field_names=['norm_rr', 'norm_rg', 'norm_rb'],
                     output_field_names=['h', 's', 'v'])

    return pd.concat([data_frame.reset_index(drop=True),
                      hsv_df.reset_index(drop=True)], axis=1)


class PolynomialRegressionModel(CoffeeMeasureCore):
    essentials = ['rr', 'rb', 'rg', 'rc']
    model_name = "polynomial_regression"

    def __dump_model__(self):
        if self.model is None:
            self.log.error("Model not available, have you train it?")
            return {}
        return {
                    'model_name': self.model_name,
                    'poly_fea': self.poly_features,
                    'poly_degree': self.poly_degree,
                    'ols_params': self.model.params.tolist(),
                    'reconstruct': 'y = dot(poly_fea(X), ols_params) * 128',
                    'note': 'The model uses the (Ordinary Least Square)OLS model to fit the best value for polynomial '
                            'features which generated with scikit-learn(sklearn). The polynomial features are quite '
                            'simple, sklearn enumerate all combination for input vector with degree D(by default is 2).'
                            ' In here, assume there are 3 dimension in given vector, which denoted to (X0, X1, X2), '
                            'then sklearn would generate following polynomial features: (1, X0, X1, X2, X0^2, X0*X1, '
                            'X0*X2, X1^2, X1*X2, X1^2). By optimizing the parameters to find the closest polynomial '
                            'function could describe the given vectors with the lowest square error compare to '
                            'observation value y.'
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
        X, y = eval_data_frame[['h', 's', 'v', 'rc']].to_numpy(), \
               eval_data_frame['value'].to_numpy() / 127
        #
        Xp = self.poly_model.transform(X)
        pred = self.model.predict(Xp)
        self.report(groundTruth=y, predicted=pred)

    def __logic__(self, hyper_params):
        self.data_frame = feature_creation(self.data_frame)
        #
        X, y = self.data_frame[['h', 's', 'v', 'rc']].to_numpy(), \
               self.data_frame['value'].to_numpy() / 127

        updated_params = {'degree': 3}
        updated_params.update(hyper_params)

        poly = PolynomialFeatures(**updated_params)
        Xp = poly.fit_transform(X)
        self.poly_model = poly
        self.poly_degree = updated_params['degree']
        self.poly_features = poly.get_feature_names()

        self.model = sm.OLS(endog=y, exog=Xp, **updated_params).fit()
        #

    def __init__(self, data_frame, log_level=30):
        self.data_frame = data_frame
        super().__init__(log_level)
        self.log.info("Essentials : {}".format(self.essentials))
