import pandas as pd
import tensorflow as tf
from joblib import load
import json
import os

from sklearn.metrics import accuracy_score


MODEL_DIR = os.environ["MODEL_DIR"]
model_file = 'cnn_model.joblib'
model_path = os.path.join(MODEL_DIR, model_file)

# Set path for the input (test data)
PROCESSED_DATA_DIR = os.environ["PROCESSED_DATA_DIR"]
test_data_path = os.path.join(PROCESSED_DATA_DIR, "testDS")
dataset_tf_element_spec = tuple((tf.TensorSpec((None, 100, 100, 58))))
# test data loaded 
test_ds = tf.data.experimental.load(test_data_path, dataset_tf_element_spec, compression = "GZIP")





# Load model
cnn_model = load(model_path)

# Compute test accuracy
score=cnn_model.evaluate(X_test,Y_test,verbose=0)
print('Test Score: ',score[0])
print('Test Accuracy: ',score[1])

# Test accuracy to JSON
test_metadata = {
    "Test Score": score[0],
    "Test Accuracy": score[1]
}


# Set output path
METRICS_DIR = os.environ["METRICS_DIR"]
test_results_file = 'test_metadata.json'
results_path = os.path.join(METRICS_DIR, test_results_file)

# Serialize and save metadata
with open(results_path, 'w') as outfile:
    json.dump(test_metadata, outfile)