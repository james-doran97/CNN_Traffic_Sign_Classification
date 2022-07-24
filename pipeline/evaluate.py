from joblib import load
import json
import os
import numpy as np



MODEL_DIR = os.environ["MODEL_DIR"]
model_file = 'cnn_model.joblib'
model_path = os.path.join(MODEL_DIR, model_file)

# Set path for the input test data
PROCESSED_DATA_DIR = os.environ["PROCESSED_DATA_DIR"]
X_test_path = os.path.join(PROCESSED_DATA_DIR, 'X_test_ds.txt')
Y_test_path = os.path.join(PROCESSED_DATA_DIR, 'Y_test_ds.txt')

# Test data loaded from directory
loaded_X_test = np.loadtxt("X_train.txt")
Y_test = np.loadtxt(Y_test_path)

# Independant test variable reshaped back into original 3D shape
X_test = loaded_X_test.reshape(loaded_X_test.shape[0], loaded_X_test.shape[1] // 100, 100)



# Load model
cnn_model = load(model_path)

# Compute test accuracy
score=cnn_model.evaluate(X_test,Y_test,verbose=0)

# Test accuracy to JSON
eval_metadata = {
    "Test Score": score[0],
    "Test Accuracy": score[1]
}


# Set output path
METRICS_DIR = os.environ["METRICS_DIR"]
eval_results_file = 'eval_metadata.json'
metrics_path = os.path.join(METRICS_DIR, eval_results_file)

# Serialize and save metadata
with open(metrics_path, 'w') as outfile:
    json.dump(eval_metadata, outfile)