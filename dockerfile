FROM python:3.9.0-slim
MAINTAINER riz_hsu

RUN mkdir /opt/makerclub
RUN mkdir /opt/makerclub/output

COPY src /opt/makerclub/src
COPY res /opt/makerclub/res

RUN cd /opt/makerclub/src

ENTRYPOINT ["python", "train.py"]