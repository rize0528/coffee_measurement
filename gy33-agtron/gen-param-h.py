# -*- coding: utf8 -*-
# Convert weights JSON to C
# Generates param.h

import json
import sys

if len(sys.argv) < 2:
    print('Usage: python3 gen-param-h.py model.json')
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
    print('''
#ifndef GY33_PARAM_H
#define MLP 1
#include <MatrixMath.h>
''')
    wx = model['reg_coef']
    for i in range(len(wx)):
        x, y = len(wx[i]), len(wx[i][0])
        print('mtx_type X{}[{}][{}] = {{'.format(i, x, y))
        for j in range(x):
            print('{%s},' % repr(wx[i][j])[1:-1])
        print('};')
    wx = model['reg_intercept']
    for i in range(len(wx)):
        x = len(wx[i])
        print('mtx_type W{}[{}] = {{{}}};'.format(i, x, repr(wx[i])[1:-1]))
    print('#endif	// GY33_PARAM_H')
else:
    print('Unsupported model: {}'.format(model['model_name']))
