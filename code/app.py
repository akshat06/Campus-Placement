from data_preprocessing.preprocessing import Preprocessing
from model_training.training import ModelTraining
import os
import pandas as pd

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