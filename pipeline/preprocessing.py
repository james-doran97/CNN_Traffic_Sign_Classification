### The following code defines the first step of the Traffic_Classifier Machine Learning pipeline, the data ingestion and processing step.

# IMPORTS 
import numpy as np
import pandas as pd
import os
import cv2
from sklearn.model_selection import train_test_split

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



# Image data converted into arrays
images = np.array(images, dtype=object)
img_class = np.array(img_class, dtype=object)

# Split into train, test, and validation datasets
X_train, X_test, Y_train, Y_test = train_test_split(images,img_class,test_size=ratio)
X_train, X_validation ,Y_train, Y_validation = train_test_split(X_train,Y_train,test_size=ratio)

# Image processing function defined, images already loaded as grayscale from ingestion, so images resized, and pixel distribution equalised, and image normalised
def img_processing(img):
    img = cv2.resize(img,(100,100))
    img=cv2.equalizeHist(img)
    img=img/255.0
    
    return img

# Mapping
X_train=np.array(list(map(img_processing,X_train)))
X_validation=np.array(list(map(img_processing,X_validation)))
X_test=np.array(list(map(img_processing,X_test)))

# Classes to binary class matrices 
Y_train=to_categorical(Y_train,class_count)
Y_validation=to_categorical(Y_validation,class_count)
Y_test=to_categorical(Y_test,class_count)

# Data save paths defined
PROCESSED_DATA_DIR = os.environ["PROCESSED_DATA_DIR"]
X_train_path = os.path.join(PROCESSED_DATA_DIR, 'X_train_ds.txt')
X_test_path = os.path.join(PROCESSED_DATA_DIR, 'X_test_ds.txt')
X_val_path = os.path.join(PROCESSED_DATA_DIR, 'X_val_ds.txt')
Y_train_path = os.path.join(PROCESSED_DATA_DIR, 'Y_train_ds.txt')
Y_test_path = os.path.join(PROCESSED_DATA_DIR, 'Y_test_ds.txt')
Y_val_path = os.path.join(PROCESSED_DATA_DIR, 'Y_val_ds.txt')

# Independant variables need to be reshaped before being saved. The savetxt method only takes 1 and 2D arrays as arguments, so 3D array needs to get reduced to a 2D array
# Data saved to processed data directory in Docker container
X_train_reshaped = X_train.reshape(X_train.shape[0] ,-1)
np.savetxt(X_train_path, X_train_reshaped)

X_test_reshaped = X_test.reshape(X_test.shape[0] ,-1)
np.savetxt(X_train_path, X_test_reshaped)

X_val_reshaped = X_validation.reshape(X_validation.shape[0] ,-1)
np.savetxt(X_train_path, X_val_reshaped)



np.savetxt(Y_train_path, Y_train)
np.savetxt(Y_test_path, Y_test)
np.savetxt(Y_val_path, Y_validation)