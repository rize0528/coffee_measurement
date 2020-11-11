FROM python:3.8.0-slim
MAINTAINER riz_hsu

RUN mkdir /opt/makerclub
RUN mkdir /opt/makerclub/output

RUN pip install -U pandas scikit-learn

COPY src /opt/makerclub/src
COPY res /opt/makerclub/res

RUN cd /opt/makerclub/src

#ENTRYPOINT ["/usr/local/bin/python", "/opt/makerclub/src/train.py --model regression --input /opt/makerclub/res/"]
