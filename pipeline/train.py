import os
from tensorflow.keras.layers import Dense , Flatten , Dropout , Conv2D , MaxPooling2D
from tensorflow.keras.models import Model , Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.callbacks import EarlyStopping
from keras.utils.np_utils import to_categorical

PROCESSED_DATA_DIR = os.environ["PROCESSED_DATA_DIR"]
train_data_file = 'train.csv'
train_data_path = os.path.join(PROCESSED_DATA_DIR, train_data_file)

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
cnn_model=model.fit_generator(datagen.flow(X_train,Y_train,batch_size=batch_size),steps_per_epoch=epoch_steps,epochs=epochs,validation_data=(X_validation,Y_validation),shuffle=1)

MODEL_DIR = os.environ["MODEL_DIR"]
model_name = 'cnn_model.joblib'
model_path = os.path.join(MODEL_DIR, model_name)
