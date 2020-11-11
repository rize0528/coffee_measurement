FROM python:3.8.0-slim
MAINTAINER riz_hsu

RUN mkdir /opt/makerclub
RUN mkdir /opt/makerclub/output

RUN pip install -U pandas scikit-learn asciichartpy

COPY src /opt/makerclub/src
COPY res /opt/makerclub/res

RUN cd /opt/makerclub/src

# Model options: [regression or mlp]
# Available options for train.py
#   |-   -m, --model:  choice between ["regression", "mlp"], model kernel type
#   |-   -i, --input:  str, filename or filepath of training data path, if a folder path are given, program would scan
#   |                       for *.csv file in that folder.
#   |-   -o, --output: str, output dirname, if the folder does not exist, program will create one.
#   |-   -p, --hyper-parameters: dictionary in string form,
#   |                            example:  -p {\"hidden_layer_sizes\":[10,10]}
#   |-   -l, --log-level: choice among ["debug", "info", "warning", "error"].

#ENTRYPOINT ["/usr/local/bin/python", "/opt/makerclub/src/train.py --model regression --input /opt/makerclub/res/"]
