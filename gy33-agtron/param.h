
#ifndef GY33_PARAM_H
#define MODEL_NAME "Polynomial Regression"
#define POLYNOMIAL_REGRESSION 1
double ols_params[20] = { 29544.3310343, -2751.72621304, 5755.30940804, -258755.181431, -5837.56178628, 19078.0220232, 16635.3992292, -12360.2256339, -39322.0425687, 757172.50357, 3197.49904353, -8113.6791448, 11580.2105429, 9189.21283875, -49522.4440641, -21078.0683088, -4085.1919163, 33623.387871, 64060.3204877, -741370.314056 };
#define CALC_POLYNOMIAL (1 * ols_params[0]) + \
    (hsvc[0] * ols_params[1]) + \
    (hsvc[1] * ols_params[2]) + \
    (hsvc[2] * ols_params[3]) + \
    (hsvc[0] * hsvc[0] * ols_params[4]) + \
    (hsvc[0] * hsvc[1] * ols_params[5]) + \
    (hsvc[0] * hsvc[2] * ols_params[6]) + \
    (hsvc[1] * hsvc[1] * ols_params[7]) + \
    (hsvc[1] * hsvc[2] * ols_params[8]) + \
    (hsvc[2] * hsvc[2] * ols_params[9]) + \
    (hsvc[0] * hsvc[0] * hsvc[0] * ols_params[10]) + \
    (hsvc[0] * hsvc[0] * hsvc[1] * ols_params[11]) + \
    (hsvc[0] * hsvc[0] * hsvc[2] * ols_params[12]) + \
    (hsvc[0] * hsvc[1] * hsvc[1] * ols_params[13]) + \
    (hsvc[0] * hsvc[1] * hsvc[2] * ols_params[14]) + \
    (hsvc[0] * hsvc[2] * hsvc[2] * ols_params[15]) + \
    (hsvc[1] * hsvc[1] * hsvc[1] * ols_params[16]) + \
    (hsvc[1] * hsvc[1] * hsvc[2] * ols_params[17]) + \
    (hsvc[1] * hsvc[2] * hsvc[2] * ols_params[18]) + \
    (hsvc[2] * hsvc[2] * hsvc[2] * ols_params[19]) + \
    0
#endif	// GY33_PARAM_H
