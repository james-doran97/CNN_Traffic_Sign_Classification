FROM jupyter/scipy-notebook
USER root

RUN pip install joblib
RUN apt-get update && apt-get install -y jq
RUN mkdir model raw_data processed_data metrics


ENV MODEL_DIR=/home/jovyan/model
ENV RAW_DATA_DIR=/home/jovyan/raw_data
ENV PROCESSED_DATA_DIR=/home/jovyan/processed_data
ENV METRICS_DIR=/home/jovyan/metrics

COPY ./pipeline/preprocessing.py ./pipeline/preprocessing.py
COPY ./pipeline/train.py ./pipeline/train.py
COPY ./pipeline/evaluate.py ./pipeline/evaluate.py
COPY ./traffic_Data ./traffic_Data
COPY ./labels/labels.csv ./labels/labels.csv
COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt
