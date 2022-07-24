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

# Hyperparameters loaded from file
with open(hyperparam_path) as h:
    hyperparameters = json.load(h)
epochs = hyperparameters["hyperparameters"][0]["epochs"]
epoch_steps = hyperparameters["hyperparameters"][0]["epch_steps"]
ratio = hyperparameters["hyperparameters"][0]["test_ratio"]
batch_size = hyperparameters["hyperparameters"][0]["batch_size"]
no_of_nodes = hyperparameters["hyperparameters"][0]["no_of_nodes"]
learning_rate = hyperparameters["hyperparameters"][0]["learning_rate"]
no_of_filters = hyperparameters["hyperparameters"][0]["no_of_filters"]

imageDimensions = (32,32,3)
size_of_filters=(5,5)
size_of_filters_2=(3,3)
size_of_pool=(2,2)

# Note: test ratio and validation ratio the same (0.1)

# Image data loaded alongside corresponding class
# Note: Image data loaded as grayscale before additional preprocessing applied in img_process function
count = 0
img_class = []
images = []
files = os.listdir(data_path)
class_count = len(files)
print("Class Count:", class_count)


for category in range (0,class_count):
        sign_img=os.listdir(data_path + "//" + str(count))
        for img in sign_img:
            sign=cv2.imread(data_path+"/"+str(count)+"/"+ img, 0)
            img_class.append(count)
            images.append(sign)
                
        count=count+1

# Lists converted to numpy arrays
images = np.array(images, dtype=object)
img_class = np.array(img_class, dtype=object)

# Split into train, test, and validation datasets
X_train, X_test, Y_train, Y_test = train_test_split(images,img_class,test_size=ratio)
X_train, X_validation ,Y_train, Y_validation = train_test_split(X_train,Y_train,test_size=ratio)

def img_processing(img):
    img = cv2.resize(img,(100,100))
    img=cv2.equalizeHist(img)
    img=img/255.0
    
    return img
