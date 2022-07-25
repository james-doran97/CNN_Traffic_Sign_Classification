import os
import json
from pipeline.preprocessing import X_validation, Y_train, Y_validation
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense , Flatten , Dropout , Conv2D , MaxPooling2D
from tensorflow.keras.models import Model , Sequential
from tensorflow.keras.optimizers import Adam
from joblib import dump

# Paths to data defined
PROCESSED_DATA_DIR = os.environ["PROCESSED_DATA_DIR"]
X_train_path = os.path.join(PROCESSED_DATA_DIR, 'X_train_ds.txt')
X_val_path = os.path.join(PROCESSED_DATA_DIR, 'X_val_ds.txt')
Y_train_path = os.path.join(PROCESSED_DATA_DIR, 'Y_train_ds.txt')
Y_val_path = os.path.join(PROCESSED_DATA_DIR, 'Y_val_ds.txt')

# training and validation data loaded
loaded_X_train = np.loadtxt("X_train.txt")
loaded_X_val = np.loadtxt("X_train.txt")
Y_train = np.loadtxt(Y_train_path)
Y_validation = np.loadtxt(Y_val_path)


# Independant training and validation variables reshaped back into original 3D shape
X_train = loaded_X_train.reshape(loaded_X_train.shape[0], loaded_X_train.shape[1] // 100, 100)
X_validation = loaded_X_val.reshape(loaded_X_val.shape[0], loaded_X_val.shape[1] // 100, 100)


# Hyperparameter path defined
hyperparam_path = ".\\conf\\conf.json"
 
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
class_count = hyperparameters["hyperparameters"][0]["class_count"]

imageDimensions = (32,32,3)
size_of_filters=(5,5)
size_of_filters_2=(3,3)
size_of_pool=(2,2)


# Model defined, 
def model():

    
    model=Sequential()
    model.add((Conv2D(no_of_filters,size_of_filters,input_shape=(100,100,1),activation='relu')))
    model.add((Conv2D(no_of_filters,size_of_filters,activation='relu')))
    model.add(MaxPooling2D(pool_size=size_of_pool))

    model.add((Conv2D(no_of_filters//2,size_of_filters_2,activation='relu')))
    model.add((Conv2D(no_of_filters//2,size_of_filters_2,activation='relu')))
    model.add(MaxPooling2D(pool_size=size_of_pool))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(no_of_nodes,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(class_count,activation='softmax'))
    model.summary()
    model.compile(Adam(learning_rate=learning_rate),loss='categorical_crossentropy',metrics=['accuracy'])
    
    return model

model=model()
cnn_model=model.fit(X_train, Y_train,batch_size=batch_size,steps_per_epoch=epoch_steps,epochs=epochs,validation_data=(X_validation, Y_validation),shuffle=1)


train_metadata = {
    'train_acc': cnn_model.history['accuracy'],
    'train_loss': cnn_model.history['loss'],
    'validation_acc': cnn_model.history['val_accuracy'],
    'validation_loss': cnn_model.history['val_loss']
}

MODEL_DIR = os.environ["MODEL_DIR"]
model_name = 'cnn_model.joblib'
model_path = os.path.join(MODEL_DIR, model_name)

dump(model_name, model_path)

METRICS_DIR = os.environ["METRICS_DIR"]
train_metrics_file = 'train_metadata.json'
metrics_path = os.path.join(METRICS_DIR, train_metrics_file)

# Serialize and save metadata
with open(metrics_path, 'w') as outfile:
    json.dump(train_metadata, outfile)
