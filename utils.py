import os,sys
#
def data_reader(path):
    with open(path, 'r') as fp:
        lines = fp.read().split('\n')
        raw_data = []

        for line in lines:
            if line.startswith('#') or len(line)==0: continue

            sp = list(filter(lambda x: len(x)>0, line.split('\t')))
            if len(sp)==3:
                benchmark = sp[1].split(' ', 1)
                if len(benchmark)<2: continue

                measure = dict([[y.strip().split(' ',1)[0] for y in x.strip().split('=')] for x in sp[2].split(',')])

                raw_data.append({
                                'note': sp[0], \
                                'bm': {
                                    'value': float(benchmark[0]),
                                    'label': benchmark[1]
                                },\
                                'measure': measure
                            })
    return raw_data
#
def feature_extractor(raw_data):
    _x = []
    _y = []
    for item in raw_data:
        benchmark_value = item.get('bm',{}).get('value', -1)
        normalized_vector = []
        for color_dim in item.get('measure', {}).items():
            _var, _vl = color_dim[0], float(color_dim[1])
            if _var == 'ct':
                _vl = _vl/8191
            elif _var == 'lux':
                _vl = _vl/31
            elif _var == 'rc':
                _vl = _vl/512
            else:
                _vl = _vl/255
            normalized_vector.append(_vl)
        _x.append(normalized_vector)
        _y.append(benchmark_value)
    
    return np.array(_x), np.array(_y)
#
