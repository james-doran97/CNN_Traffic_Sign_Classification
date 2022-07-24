### The following code defines the first step of the Traffic_Classifier Machine Learning pipeline, the data ingestion and processing step.

# IMPORTS 
import tensorflow as tf
import numpy as np
import pandas as pd
import os
from tensorflow.keras.layers import Dense , Flatten , Dropout , Conv2D , MaxPooling2D
from tensorflow.keras.models import Model , Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.callbacks import EarlyStopping
from keras.utils.np_utils import to_categorical
import cv2
from sklearn.model_selection import train_test_split
import json
from keras.preprocessing.image import ImageDataGenerator

# Image data, labels, and hyperparameters path 
labels_path = ".\\labels\\labels.csv"
data_path = ".\\traffic_Data"


# Labels read
labels = pd.read_csv(labels_path)

batch_size = 10
ratio = 0.1
# Note: test ratio and validation ratio the same (0.1)

# Image data loaded alongside corresponding class
# Note: Image data loaded as grayscale before additional preprocessing applied in img_process function
count = 0
img_class = []
images = []
files = os.listdir(data_path)
class_count = len(files)


for category in range (0,class_count):
        sign_img=os.listdir(data_path + "//" + str(count))
        for img in sign_img:
            sign=cv2.imread(data_path+"/"+str(count)+"/"+ img, 0)
            img_class.append(count)
            images.append(sign)
                
        count=count+1

print(img_class)

def img_processing(img):
    img = cv2.resize(img,(100,100))
    img=cv2.equalizeHist(img)
    img=img/255.0
    
    return img
# Split into train, test, and validation datasets
X_train, X_test, Y_train, Y_test = train_test_split(images,img_class,test_size=ratio)
X_train, X_validation ,Y_train, Y_validation = train_test_split(X_train,Y_train,test_size=ratio)
X_train=X_train.reshape(X_train.shape[0],100,100,1)
X_validation=X_validation.reshape(X_validation.shape[0],100,100,1)
X_test=X_test.reshape(X_test.shape[0],100,100,1)
datagen=ImageDataGenerator(width_shift_range=0.1,height_shift_range=0.1,zoom_range=0.2,shear_range=0.1,rotation_range=10)
datagen.fit(X_train)
batches=datagen.flow(X_train,Y_train,10)
X_batch,Y_batch=next(batches)


Y_train=to_categorical(Y_train,class_count)
Y_validation=to_categorical(Y_validation,class_count)
Y_test=to_categorical(Y_test,class_count)

AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = tf.data.Dataset.from_tensor_slices(X_train, Y_train)
test_ds = tf.data.Dataset.from_tensor_slices(X_test, Y_test)
val_ds = tf.data.Dataset.from_tensor_slices(X_validation, Y_validation)

train_ds = train_ds.map(lambda images,
                      img_class: img_processing(X_train, Y_train), num_parallel_calls = AUTOTUNE)
test_ds = test_ds.map(lambda images,
                      img_class: img_processing(X_test, Y_test), num_parallel_calls = AUTOTUNE)
val_ds = val_ds.map(lambda images,
                      img_class: img_processing(X_validation, Y_validation), num_parallel_calls = AUTOTUNE)




# Batch data
train_ds = train_ds.batch(batch_size)
test_ds = test_ds.batch(batch_size)
val_ds = val_ds.batch(batch_size)
# Fetches data in background while model is training
train_ds = train_ds.prefetch(buffer_size = AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size = AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size = AUTOTUNE)

PROCESSED_DATA_DIR = os.environ["PROCESSED_DATA_DIR"]
train_path = os.path.join(PROCESSED_DATA_DIR, 'trainDS')
test_path = os.path.join(PROCESSED_DATA_DIR, 'testDS')
val_path = os.path.join(PROCESSED_DATA_DIR, 'valDS')

tf.data.experimental.save(train_ds, train_path, compression = "GZIP")
tf.data.experimental.save(test_ds, train_path, compression = "GZIP")
tf.data.experimental.save(val_ds, train_path, compression = "GZIP")



