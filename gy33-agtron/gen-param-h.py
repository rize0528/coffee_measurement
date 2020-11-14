# -*- coding: utf8 -*-
# Convert weights JSON to C
# Generates param.h

import json
import sys
from itertools import combinations_with_replacement

if len(sys.argv) < 2:
    print('Usage: python3 gen-param-h.py model.json > param.h')
    sys.exit(1)

model = json.load(open(sys.argv[1]))

if model['model_name'] == 'linear_regression':
    print('''
#ifndef GY33_PARAM_H
#define LINEAR_REGRESSION 1
#define X_R {}
#define X_G {}
#define X_B {}
#define X_C {}
'''.format(*model['reg_coef']))
    print('#define BIAS {}'.format(model['reg_intercept']))
    print('#endif	// GY33_PARAM_H')
elif model['model_name'] == 'MLP':
    max_dim = 0
    print('''
#ifndef GY33_PARAM_H
#define MLP 1
#include <MatrixMath.h>
''')
    wx = model['mlp_weights']
    for i in range(len(wx)):
        x, y = len(wx[i]), len(wx[i][0])
        max_dim = max(max_dim, max(x, y))
        print('mtx_type X{}[{}][{}] = {{'.format(i, x, y))
        for j in range(x):
            print('{%s},' % repr(wx[i][j])[1:-1])
        print('};')
    wx = model['mlp_bias']
    for i in range(len(wx)):
        x = len(wx[i])
        print('mtx_type W{}[{}] = {{{}}};'.format(i, x, repr(wx[i])[1:-1]))
    print('#define MAX_DIM {}'.format(max_dim))
    print('#endif	// GY33_PARAM_H')
elif model['model_name'] == 'polynomial_regression':
    print('''
#ifndef GY33_PARAM_H
#define POLYNOMIAL_REGRESSION 1
double ols_params[{}] = {{ {} }};
'''.format(len(model['ols_params']), ', '.join(map(str, model['ols_params']))))
    print('#define CALC_POLYNOMIAL (1 * ols_params[0]) + \\')
    noms = []
    for x in range(4):
        noms.append('hsvc[{}]'.format(x))
    i = 1
    for deg in range(1, model['poly_degree'] + 1):
        for nominal in combinations_with_replacement(noms, deg):
            print('    ({} * ols_params[{}]) + \\'.format(' * '.join(nominal), i))
            i += 1
    print('    0')
    print('#endif	// GY33_PARAM_H')
else:
    print('Unsupported model: {}'.format(model['model_name']))
