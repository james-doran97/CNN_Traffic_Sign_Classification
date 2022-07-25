# CNN_Traffic_Sign_Classification

CNN Classification model pipeline code, along with corresponding DockerFile.

ML Pipeline split into 4 steps, Preprocessing, Training, Evaluation, Deployment.
This pipeline is containerised and run through a Jenkins localhost. 



## Preprocesssing 
- Images initially read in through opencv as grayscale
- Then split into test, train, and validation datasets
- Further processing takes place, resizing, normalisation, and image pixel distribution equalisation
- Data then saved into local directory inside docker container, for use in training

## Training
- Training and Validation Data and hyperparameters loaded, and CNN model defined
- Model trained, and saved into local directory.
- Model training metrics pushed to local directory

## Evaluation
- Model loaded, along with test data
- Model evaluated, with evaluation metrics saved to json pushed to local directory

## Deploy
- Flask used to host cnn_model as a REST API endpoint
- Class defined to read in data from request, process, and sent to model for prediction
- Prediction payload returns prediction and confidence value

