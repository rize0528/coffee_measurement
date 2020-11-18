import os
import sys
import logging
import colorsys
import importlib
import pandas as pd

sys.path.append('/opt/makerclub/src')

def rgb2hsv(data_frame: pd.DataFrame,
            field_names=None,
            output_field_names=None) -> pd.DataFrame:
    # Note: field_names should in correct orders: r->g->b
    #
    if field_names is None:
        field_names = ['rr', 'rg', 'rb']
    if output_field_names is None:
        output_field_names = ['h', 's', 'v']
    #
    _hsv_meta = list(map(lambda x: colorsys.rgb_to_hsv(x[1].get(field_names[0]),
                                                       x[1].get(field_names[1]),
                                                       x[1].get(field_names[2])),
                         data_frame[field_names].iterrows()))
    return pd.DataFrame(_hsv_meta, columns=output_field_names)


def module_scanner(scan_path: str, scan_posfix: str) -> list:
    candidate_models = list(filter(lambda x: x.endswith(scan_posfix), os.listdir(scan_path)))
    valid_model = []
    for _model_name in candidate_models:
        try:
            _package_name = _model_name.rstrip(".py")
            m = importlib.import_module(_package_name, _package_name)
            if hasattr(m, 'Model'):
                valid_model.append(_package_name)
        except Exception as e:
            logging.warning("Error message: {}".format(str(e)))
            logging.warning("Unable to load module: {} from {}".format(_package_name, _model_name))
            logging.warning(f"Given scan_path={scan_path}, scan_posfix={scan_posfix}")
    return valid_model


