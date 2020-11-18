# -*- coding: utf8 -*-
# Convert weights JSON to C
# Generates param.h

import os
import sys
import json
import argparse
from itertools import combinations_with_replacement


def param_generator(model, output_stream):
    if model['model_name'] == "linear_regression":
        output_stream.write('''
#ifndef GY33_PARAM_H
#define MODEL_NAME "Linear Regression"
#define LINEAR_REGRESSION 1
#define X_R {}
#define X_G {}
#define X_B {}\n'''.format(*model['reg_coef']))
        output_stream.write('#define BIAS {}\n'.format(model['reg_intercept']))
        output_stream.write('#endif	// GY33_PARAM_H\n')
    elif model['model_name'] == "MLP":
        max_dim = 0
        output_stream.write('''
#ifndef GY33_PARAM_H
#define MODEL_NAME "MLP"
#define MLP 1
#include <MatrixMath.h>\n''')
        wx = model['mlp_weights']
        for i in range(len(wx)):
            x, y = len(wx[i]), len(wx[i][0])
            max_dim = max(max_dim, max(x, y))
            output_stream.write('mtx_type X{}[{}][{}] = {{\n'.format(i, x, y))
            for j in range(x):
                output_stream.write('{%s},\n' % repr(wx[i][j])[1:-1])
            output_stream.write('};\n')
        wx = model['mlp_bias']
        for i in range(len(wx)):
            x = len(wx[i])
            output_stream.write('mtx_type W{}[{}] = {{{}}};\n'.format(i, x, repr(wx[i])[1:-1]))
        output_stream.write('#define MAX_DIM {}\n'.format(max_dim))
        output_stream.write('#endif	// GY33_PARAM_H\n')
    elif model['model_name'] == "polynomial_regression":
        output_stream.write('''
#ifndef GY33_PARAM_H
#define MODEL_NAME "Polynomial Regression"
#define POLYNOMIAL_REGRESSION 1
double ols_params[{}] = {{ {} }};\n'''.format(len(model['ols_params']), ', '.join(map(str, model['ols_params']))))
        output_stream.write('#define CALC_POLYNOMIAL (1 * ols_params[0]) + \\\n')
        noms = []
        for x in range(4):
            noms.append('hsvc[{}]'.format(x))
        i = 1
        for deg in range(1, model['poly_degree'] + 1):
            for nominal in combinations_with_replacement(noms, deg):
                output_stream.write('    ({} * ols_params[{}]) + \\\n'.format(' * '.join(nominal), i))
                i += 1
        output_stream.write('    0\n')
        output_stream.write('#endif	// GY33_PARAM_H\n')
    else:
        print('[Error] Unsupported model: {}, program halt.'.format(model['model_name']))


def main():
    parser = argparse.ArgumentParser(description='Usage: python3 gen-param-h.py model.json > param.h')
    parser.add_argument("file", help='model json file path')
    args = parser.parse_args()

    if os.path.exists(os.path.realpath(args.file)):
        model = json.load(open(os.path.realpath(args.file)))
        param_generator(model, sys.stdout)
    else:
        print('[Error] Model')


if __name__ == "__main__":
    sys.exit(main())
