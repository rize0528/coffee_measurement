
#ifndef GY33_PARAM_H
#define MODEL_NAME "Polynomial Regression"
#define POLYNOMIAL_REGRESSION 1
double ols_params[20] = { 99.46220693744091, -155.89858878364157, 264.58783648141366, -917.2265211494225, 171.86242204736487, -272.7821083715194, 735.5748264134206, 265.48253606878666, -1675.5717025507306, 2960.479748284714, -58.982604961248356, 214.91377903261593, -417.3899079367866, -232.80400173336207, 615.4160289200412, -880.6200285328957, 119.32743674490902, -762.6306344113709, 2715.0125185964444, -3261.3974257952013 };
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
