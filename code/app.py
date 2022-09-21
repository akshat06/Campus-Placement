from data_preprocessing.preprocessing import Preprocessing
from model_training.training import ModelTraining
from model_prediction.prediction import Predicition
import os
import pandas as pd
import logging

logging_str = "[%(asctime)s: %(levelname)s: %(module)s] %(message)s"
log_dir = "code\logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename= os.path.join(log_dir,"running_logs.log"),level=logging.INFO, format=logging_str, filemode="a")

"""Data Preprocessing
"""
train_file_path = os.path.join("code","dataset","train.csv")
print(train_file_path)
pre_process = Preprocessing(filename=train_file_path)
pre_process.preprocessing_flow()

"""Model Training
"""
model_train = ModelTraining()
model_train.training_flow()


"""Model Prediction
"""
model = os.path.join("code","model_file","best_model.pkl")
pred_data = [0,56.0,1,52.0,1,1,52.0,1,0,66.0,1,59.43,288655.4054054054]
model_pred = Predicition(model_file=model, data=pred_data)
if model_pred == 0:
    logging.info("\n\nPrediction-->> Student will not be placed!")
else:
    logging.info("\n\nPrediction-->> Student will get placed!!")