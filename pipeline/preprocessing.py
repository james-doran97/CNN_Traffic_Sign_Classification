### The following code defines the first step of the Traffic_Classifier Machine Learning pipeline, the data ingestion and processing step.

# IMPORTS 
import tensorflow as tf
import numpy as np
import pandas as pd
import mlflow
from mlflow.pipelines import Pipeline
from mlflow.pyfunc import PyFuncModel
import os
from tensorflow.keras.layers import Dense , Flatten , Dropout , Conv2D , MaxPooling2D
from tensorflow.keras.models import Model , Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.callbacks import EarlyStopping
from keras.utils.np_utils import to_categorical
import cv2
from sklearn.model_selection import train_test_split
import pickle
import random
from keras.preprocessing.image import ImageDataGenerator
import json

# Image data, labels, and hyperparameters path 
labels_path = ".\\labels\\labels.csv"
data_path = ".\\traffic_Data"
hyperparam_path = ".\\conf\\conf.json"

# Labels read
labels = pd.read_csv(labels_path)