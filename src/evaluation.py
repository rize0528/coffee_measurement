import os, sys
import json
import logging
import argparse
import numpy as np
import pandas as pd
import importlib
import asciichartpy
import plotly.graph_objects as go
from itertools import chain
from datetime import datetime
from utils import module_scanner
from tqdm import tqdm

SCRIPT_WD = os.path.dirname(os.path.realpath(__file__))
DEFAULT_TRAINING_DATA = os.path.join(SCRIPT_WD, "../res/training_data_gy33_v2.csv")
SUPPORTED_MODELS = {
    'linear_regression': ['regression', 'reg'],
    'mlp_regressor': ['mlp_regressor', 'mlp'],
    'poly_regression': ['poly', 'poly_reg']
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', default=DEFAULT_TRAINING_DATA, type=str, required=True)
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

    logging.basicConfig(level=log_level, format='[%(asctime)-15s][%(levelname)s][%(filename)s] %(message)s')

    try:
        hyperp = json.loads(args.hyper_parameters)
    except Exception as e:
        logging.error("Unable to parse given hyper-parameter string. Program halt...")
        raise str(e)

    # Create output folder if not exist.
    os.makedirs(os.path.realpath(args.output), exist_ok=True)

    # Import selected models
    scanned_model = module_scanner(SCRIPT_WD, "Model.py")
    models = {}
    logging.debug('{} models found, which are: [{}]'.format(len(scanned_model), ",".join(scanned_model)))
    for _model in scanned_model:
        models[_model] = importlib.import_module(_model, _model)

    #
    loaded_df = []
    loaded_name = []
    if os.path.isdir(args.input):
        # Load all csv files
        _files = list(map(lambda y: os.path.join(args.input, y),
                          filter(lambda x: x.endswith('.csv'), os.listdir(args.input))))
        _loaded, _failed = 0, 0
        for _file in _files:
            try:
                loaded_df.append(pd.read_csv(_file))
                loaded_name.append(os.path.basename(_file))
                _loaded += 1
            except Exception as e:
                logging.warning('"{0}" can\'t be loaded.'.format(_file))
                logging.warning(str(e))
                _failed += 1
        logging.info('{0} file(s) were loaded.'.format(_loaded))
        if _failed > 0:
            logging.info('{0} file(s) were failed to load.'.format(_failed))
    else:
        logging.warning("Invalid file type. aborted")
        return 1
    #
    df_count = len(loaded_df)

    if args.model in SUPPORTED_MODELS['linear_regression']:
        model_name = 'RegressionModel'
    elif args.model in SUPPORTED_MODELS['mlp_regressor']:
        model_name = 'MLPModel'
    elif args.model in SUPPORTED_MODELS['poly_regression']:
        model_name = 'PolynomialRegressionModel'
    else:
        logging.error("No such model")
        sys.exit(1)

    reports = {}
    for idx, target_df in enumerate(loaded_df):
        _name = loaded_name[idx]
        df_train_partial = pd.concat([loaded_df[_] for _ in range(df_count) if _ != idx], axis=0)
        df_train_all = pd.concat(loaded_df)

        rm = models[model_name].Model(df_train_partial, {'log_level': log_level})
        rm.train(hyper_params=hyperp, apply_evaluation=False)
        pred_value_partial, target_partial = rm.predict(target_df)
        assessment_string = rm.report_string(target_partial, pred_value_partial)

        error_partial = pred_value_partial - target_partial

        rm_all = models[model_name].Model(df_train_all, {'log_level': log_level})
        rm_all.train(hyper_params=hyperp, apply_evaluation=False)
        pred_value_all, target_all = rm_all.predict(target_df)
        error_all = pred_value_all - target_all

        name = _name.split('.')[0].replace("gt_event_", "")

        reports[name] = {
            'Outperform': np.mean(np.abs(error_partial)) < np.mean(np.abs(error_partial)),
            'A_mad': np.mean(np.abs(error_all) * 128),
            'A_error': error_all*128,
            'P_mad': np.mean(np.abs(error_partial) * 128),
            'P_error': error_partial*128,
            'D_mad': np.abs(np.mean(np.abs(error_all) * 128) - np.mean(np.abs(error_partial) * 128)),
            'A_maxmin': [np.max(error_all) * 128, np.min(error_all) * 128],
            'P_maxmin': [np.max(error_partial) * 128, np.min(error_partial) * 128],
        }

    # Metrics
    bestScore = min([x['D_mad'] for x in reports.values()])
    worstScore = max([x['D_mad'] for x in reports.values()])
    avgScore = np.mean([x['D_mad'] for x in reports.values()])
    Scores = sorted([x['D_mad'] for x in reports.values()])
    All_a_mad = sorted([x['A_mad'] for x in reports.values()])
    subjects = len(reports.keys())

    for idx, target_df in enumerate(loaded_df):
        name = _name.split('.')[0].replace("gt_event_", "")
        output_filepath = os.path.join(args.output, name,
                                       'report_{}_{}.md'.format(name,
                                                                args.model))
        os.makedirs(os.path.join(args.output, name), exist_ok=True)
        _report = reports[name]
        data_size = target_df.shape[0]
        total = 35
        overall_score = (1 - _report['D_mad'] / np.sum(Scores)) * 40 + 60

        # Ascii Chart config
        config = {
            'height': 8,
            'display_width': 100
        }

        # Plotly setting
        plotly_cfg = {
            'width': 400,
            'height': 100,
            'margin': {'l':5, 'r':5, 'b':5, 't':40, 'pad':0}
        }

        # Ranks
        rank = Scores.index(_report['D_mad']) + 1
        amad_rank = All_a_mad.index(_report['A_mad']) + 1

        with open(output_filepath, 'w') as wp:
            wp.write(f"# MakerClub 咖啡粉偵測儀活動成績單\n")
            wp.write("* 活動時間: 2020/11/18\n")
            wp.write(f"* 參加人名稱: **{name.capitalize()}**\n")
            wp.write(f"* 模型名稱: **{args.model}**\n")
            wp.write("## 資料能力：\n")
            wp.write(f"> 資料分數:{overall_score}\n> 排名:{rank}/{total} (*1)\n")
            wp.write("### 貢獻訓練資料量:\n")
            wp.write("> \t[{0}{1}]-({2}/{3})\n".format('★' * data_size, '☆' * (total - data_size), data_size, total))
            wp.write("### 資料對模型的乖離排名:\n")
            wp.write("> \t[{0}{1}]-({2}/{3}) (*2)\n".format('★' * amad_rank, '☆' * (subjects - amad_rank), amad_rank, subjects))
            wp.write("### 模型誤差圖(*3):\n")
            _fig = go.Figure(data=[go.Scatter(y=_report['A_error'])])
            _fig.update_layout(title="Model-All v.s. 你的資料", **plotly_cfg)
            _fig.to_image(format="png", engine="kaleido")
            _figPath = os.path.join(args.output, name, '001.png')
            _fig.write_image(_figPath)
            wp.write("> ![001](001.png)\t")
            _fig = go.Figure(data=[go.Scatter(y=_report['P_error'])])
            _fig.update_layout(title="Model-User v.s. 你的資料", **plotly_cfg)
            _fig.to_image(format="png", engine="kaleido")
            _figPath = os.path.join(args.output, name, '002.png')
            _fig.write_image(_figPath)
            wp.write("|![002](002.png)\n")
            #_chart2 = asciichartpy.plot(series=[_report['P_error'].tolist()], cfg=config)
            #wp.write("{}\n".format(_chart2))

            wp.write("## 附錄\n")
            wp.write("* 模型評估說明：\n")
            wp.write("  - 評估時，將對每位學員個別製作兩個模型，分別為：全體參加學員的資料訓練的模型(**Model-All**)與僅不使用你的資料去訓練的模型(**Model-User**)。\n")
            wp.write("  - 假設**Model-All**對你貢獻的資料的平均誤差是6，而**Model-User**的平均誤差是11(大於6)，就表示你的資料對於模型的泛化能力有較高的機會提供了正向貢獻。\n")
            wp.write("```\n")
            wp.write("(*1) : 資料分數為你收集的資料對於整體模型的影響程度，越高分表示影響程度越高。\n")
            wp.write("(*2) : 乖離排名的計算是由上述兩個模型分別進行預測，利用所得到的平均絕對誤差的差值做排名。\n")
            wp.write("(*3) : 誤差值是模型對於你的資料所預測出來的數值與CM-100所測得的誤差。\n")
            wp.write("```\n")
        # Build

        break

    return 0


if __name__ == "__main__":
    sys.exit(main())
