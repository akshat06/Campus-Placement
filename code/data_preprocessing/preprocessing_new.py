from cProfile import label
from fileinput import filename
import pandas as pd
import logging
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import pickle


logging_str = "[%(asctime)s: %(levelname)s: %(module)s] %(message)s"
log_dir = "code\logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename= os.path.join(log_dir,"running_logs.log"),level=logging.INFO, format=logging_str, filemode="a")

class Preprocessing():
    def __init__(self, filename) -> None:
        self.filename = filename

    def get_data(self):
        try:
            df = pd.read_csv(self.filename)
            logging.info(f"Filename is : {self.filename}")
            logging.info(f"This is Actual Dataframe: {df.head()}")
            logging.info(f"Shape of data is: {df.shape}")
            return df
        except Exception as e:
            return logging.exception(f"Exception-->> {e}")

    def fill_missing_values(self, df, column, fill_val):
        try:
            df[column].fillna(value=fill_val, inplace=True)
            logging.info(f"Missing Value imputation with mode done sucessfully")
            return df
        except Exception as e:
            logging.exception(f"Exception-->> {e}")

    def outlier_detection(self, df, plt_dir):
        try:
            os.makedirs(plt_dir, exist_ok=True)
            logging.info("Outlier detection started!!")
            plt.figure(figsize=(15,10))

            ax = plt.subplot(231)
            plt.boxplot(df['ssc_p'])
            ax.set_title('SSC Percentage')

            ax = plt.subplot(232)
            plt.boxplot(df['hsc_p'])
            ax.set_title('HSC Percentage')

            ax = plt.subplot(233)
            plt.boxplot(df['degree_p'])
            ax.set_title('Degree Percentage')

            ax = plt.subplot(234)
            plt.boxplot(df['etest_p'])
            ax.set_title('Etest Percentage')

            ax = plt.subplot(235)
            plt.boxplot(df['mba_p'])
            ax.set_title('MBA Percentage')

            plt_path = os.path.join(plt_dir,"outlier_detection"+".jpg")
            plt.savefig(plt_path)
            logging.info(f"Outlier detection succesfully done and plot save at:{plt_path}")
        except Exception as e:
            logging.exception(f"Exception occured at outlier detection: {e}")

    def outlier_handling(self, df, col_name):
        try:
            logging.info(f"Handling of Outliers Started and shape of dataframe is {df.shape}")
            Q1 = df[col_name].quantile(0.25)
            Q3 = df[col_name].quantile(0.75)
            logging.info(f"Q1 : {Q1} and Q3: {Q3}")
            IQR = Q3-Q1
            logging.info(f"InterQuantileRange IQR : {IQR}")
            logging.info("Removing the rows containing outliers!!")
            filter_outlier = (df[col_name]>= Q1 - 1.5*IQR) & (df[col_name]<=Q3 + 1.5*IQR)
            df_filtered = df.loc[filter_outlier]
            logging.info(f"Outliers removed sucessfully and shape of dataframe is : {df_filtered.shape}")
            return df_filtered
        except Exception as e:
            logging.exception(f"Exception occured at Handling outliers: {e}")

    def drop_column(self, df, column_list):
        try:
            logging.info(f"Dropping the columns: {column_list}")
            df.drop(column_list, axis=1, inplace=True)
            logging.info(f"Successfully dropped columns: {column_list}")
            return df
        except Exception as e:
            logging.log(f"Exception occurred in dropping the column: {e}")

    def split_features_labels(self, df, target_col):
        try:
            logging.info(f"Splitting the Dataframe into features and labels where Target column is {target_col}")
            X = df.drop(columns=[target_col], axis=1)
            y = df[target_col]
            logging.info(f"Splitted Features and Target column successfully!!")
            return X, y
        except Exception as e:
            logging.exception(f"Exception Occurs-->> {e}")

    def split_train_test(self,X,y):
        try:
            logging.info("Splitting The data into Train and Test Split")
            X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=21)
            logging.info(f"Splitted Data Sucessfully X_train:{X_train.shape}, X_test:{X_test.shape},y_train:{y_train.shape}, y_test: {y_test.shape}")
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.exception(f"Exception Occurs-->> {e}")

    def label_encoding(self, df, col_list):
        try:
            logging.info(f"Performing Label Encoding on : {col_list}")
            label_encoder = LabelEncoder()
            for col in col_list:
                df[col] = label_encoder.fit_transform(df[col])
            logging.info("Label Encoding done sucessfully!!")
            return df
        except Exception as e:
            logging.exception(f"Exception occured at Label Encoding: {e}")
    
    def one_hot_encoding(self, df, col_name, prefix_name):
        try:
            dummy_col = pd.get_dummies(df[col_name], prefix=prefix_name)
            df_encoded = pd.concat([df, dummy_col], axis=1)
            logging.info(f"One hot encoding for {col_name} done sucessfully and dataset looks like:\n {df_encoded.head()}")
            return df_encoded
        except Exception as e:
            logging.exception(f"Exception occurred at one hot encoding: {e}")

    def save_csv(self, df, dir_name, file_name):
        try:
            logging.info("Saving the Clean csv.")
            os.makedirs(dir_name, exist_ok=True)
            filepath = os.path.join(dir_name,file_name)
            df.to_csv(filepath,index=False)
            logging.info(f"Saving the clean csv to {filepath}")
        except Exception as e:
            logging.exception(f"Exception in saving csv -->> {e}")

    def preprocessing_flow(self):
        logging.info("************************** Started Preprocessing Pipeline *****************************")
        # 1. Get Data
        df = self.get_data()
        # 2. Drop sl_no
        df = self.drop_column(df, ["sl_no"])
        # 3. fill salary missing values with 0
        df = self.fill_missing_values(df, "salary", 0)
        # 4. remove ssc_b, hsc_b
        df = self.drop_column(df, ["ssc_b","hsc_b"])
        # 5. outlier detection
        plt_dir = os.path.join("code","plots")
        self.outlier_detection(df, plt_dir)
        # 6. outlier handling
        df = self.outlier_handling(df, 'hsc_p')
        # 7. do label encoding for ['workex', 'specialisation', 'status']
        col_list = ['workex', 'specialisation', 'status']
        df = self.label_encoding(df, col_list)
        # 8. do one hot encoding for ['hsc_s','degree_t']
        df = self.one_hot_encoding(df = df, col_name='hsc_s', prefix_name='hsc')
        df = self.one_hot_encoding(df = df, col_name='degree_t', prefix_name='degree')
        # 9. drop columns ['workex', 'specialisation', 'status','hsc_s','degree_t','salary]
        df = self.drop_column(df, column_list=['hsc_s','degree_t','salary'])
        # 10. save the csv file
        dir_name = os.path.join("code","clean_data")
        file_name = "preprocessed_data.csv"
        self.save_csv(df, dir_name, file_name)
        logging.info("************************** Preprocessing Pipeline Ended *****************************")