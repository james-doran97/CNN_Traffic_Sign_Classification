import os
import json
import tensorflow as tf
from tensorflow.keras.layers import Dense , Flatten , Dropout , Conv2D , MaxPooling2D
from tensorflow.keras.models import Model , Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.callbacks import EarlyStopping
from keras.utils.np_utils import to_categorical
from joblib import dump
from sklearn.model_selection import StratifiedKFold, cross_val_score

PROCESSED_DATA_DIR = os.environ["PROCESSED_DATA_DIR"]
train_data_path = os.path.join(PROCESSED_DATA_DIR, "trainDS")
val_data_path = os.path.join(PROCESSED_DATA_DIR, "valDS")
hyperparam_path = ".\\conf\\conf.json"
# element spec, based on dimensions of image, and number of classes (58)
dataset_tf_element_spec = tuple((tf.TensorSpec((None, 100, 100, 58))))
# training and validation data loaded 
train_ds = tf.data.experimental.load(train_data_path, dataset_tf_element_spec, compression = "GZIP")
val_ds = tf.data.experimental.load(val_data_path, dataset_tf_element_spec, compression = "GZIP")
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
cnn_model=model.fit(train_ds,batch_size=batch_size,steps_per_epoch=epoch_steps,epochs=epochs,validation_data=val_ds,shuffle=1)

cv = StratifiedKFold(n_splits=3) 
val_model = cross_val_score(cnn_model, train_ds cv=cv).mean()

train_metadata = {
    'validation_acc': val_logit
}

MODEL_DIR = os.environ["MODEL_DIR"]
model_name = 'cnn_model.joblib'
model_path = os.path.join(MODEL_DIR, model_name)

dump(model_name, model_path)

RESULTS_DIR = os.environ["RESULTS_DIR"]
train_results_file = 'train_metadata.json'
results_path = os.path.join(RESULTS_DIR, train_results_file)

# Serialize and save metadata
with open(results_path, 'w') as outfile:
    json.dump(train_metadata, outfile)
