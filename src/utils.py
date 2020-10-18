import pandas as pd
import colorsys


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
    _hsv_meta = list(map(lambda x: colorsys.rgb_to_hsv(x[1].rr, x[1].rg, x[1].rb),
                         data_frame[field_names].iterrows()))
    return pd.DataFrame(_hsv_meta, columns=output_field_names)
