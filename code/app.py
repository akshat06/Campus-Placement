from data_preprocessing.preprocessing import Preprocessing
import os
import pandas as pd

train_file_path = os.path.join("code","dataset","train.csv")
print(train_file_path)
pre_process = Preprocessing(filename=train_file_path)
pre_process.preprocessing_flow()

# pd.read_csv(train_file_path)

# print(os.path.exists(train_file_path))