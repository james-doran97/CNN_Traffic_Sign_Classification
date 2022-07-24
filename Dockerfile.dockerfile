FROM jupyter/scipy-notebook
RUN pip install joblib


USER root
RUN apt-get update && apt-get install -y jq

RUN mkdir model raw_data processed_data metrics

ENV MODEL_DIR=/home/jovyan/model
ENV RAW_DATA_DIR=/home/jovyan/raw_data
ENV PROCESSED_DATA_DIR=/home/jovyan/processed_data
ENV METRICS_DIR=/home/jovyan/metrics

COPY preprocessing.py ./pipeline/preprocessing.py
COPY train.py ./pipeline/train.py
COPY evaluate.py ./pipeline/test.py
COPY traffic_Data
COPY labels.csv ./labels/labels.csv