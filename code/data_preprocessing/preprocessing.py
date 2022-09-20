from fileinput import filename
import pandas as pd
import logging
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

logging_str = "[%(asctime)s: %(levelname)s: %(module)s] %(message)s"
log_dir = "code\logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename= os.path.join(log_dir,"running_logs.log"),level=logging.INFO, format=logging_str, filemode="a")

class Preprocessing():
    def __init__(self,filename):
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

    def fill_missing_values(self, df, column, imputation="mean"):
        try:
            logging.info("Filling Missing Values")
            if imputation == "mean":
                logging.info(f"Filling Missing Values using Mean Imputation")
                fill_val = df[column].mean()
                df[column].fillna(value=fill_val, inplace=True)
                logging.info(f"Missing Value imputation with mean done sucessfully")
            elif imputation == "median":
                logging.info(f"Filling Missing Values using Median Imputation")
                fill_val = df[column].median()
                df[column].fillna(value=fill_val, inplace=True)
                logging.info(f"Missing Value imputation with Median done sucessfully")
            elif imputation == "mode":
                logging.info(f"Filling Missing Values using Mode Imputation")
                fill_val = df[column].mode()
                df[column].fillna(value=fill_val, inplace=True)
                logging.info(f"Missing Value imputation with mode done sucessfully")
            else:
                logging.log(f"Imputation method is Wrong: {imputation}")
        except Exception as e:
            logging.exception(f"Exception-->> {e}")

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
            X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
            logging.info(f"Splitted Data Sucessfully X_train:{X_train.shape}, X_test:{X_test.shape},y_train:{y_train.shape}, y_test: {y_test.shape}")
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.exception(f"Exception Occurs-->> {e}")

    # def convert_categorical_data(self, df, column, to_replace, values):
    #     """list of categorical columns is:
    #     ['ssc_b','hsc_b','hsc_s','degree_t','workex','Specialization','status']

    #     Args:
    #         df (_type_): _description_
    #         column (_type_): _description_
    #         to_replace (_type_): _description_
    #         values (_type_): _description_
    #     """
    #     try:
    #         logging.info(f"Converting Categorical Data {column} into Numerical!!")
    #         df[column].replace(to_replace=to_replace, value = values, inplace = True)
    #     except Exception as e:
    #         logging.exception(f"Exception -->> {e}")
    
    def standardize_data(self, X_train, X_test, file_name):
        try:
            logging.info("Scaling the data using Standard Scaler!")
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            logging.info(f"shape of X_train is: {X_train.shape}")
            # logging.info(f"Column names of X_train:{X_train.columns}")
            os.makedirs("code\model_file", exist_ok=True)
            filepath = os.path.join("code","model_file",file_name)
            pickle.dump(scaler,open(filepath,'wb'))
            logging.info(f"Scaling Done and saved the model file {filepath}")
        except Exception as e:
            logging.exception("Exception-->> {e}")

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
        try:
            logging.info(f"\n\n************ Starting the Preprocessing Pipeline!! ************")
            # 1. Getting The data
            df = self.get_data()

            # Dropping the serial_no column as it provides no valuable information
            logging.info("Dropping serial_no column from dataframe")
            df.drop(columns=['sl_no'], axis=1, inplace=True)
            
            # 2. Filling the missing value
            self.fill_missing_values(df, column ='salary', imputation="mean")
            
           
            
            # 3. Convert categorical data to numerical
            # cat_col_list = ['ssc_b','hsc_b','hsc_s','degree_t','workex','Specialization','status']
            # self.convert_categorical_data(self, df, column, to_replace, values)
            df['ssc_b'].replace(to_replace=['Central','Others'],value=[1,0], inplace=True)
            df['hsc_b'].replace(to_replace=['Central','Others'], value = [1,0], inplace=True)
            df['hsc_s'].replace(to_replace=['Commerce','Science','Arts'], value=[0,1,2], inplace=True)
            df['degree_t'].replace(to_replace=['Comm&Mgmt','Sci&Tech','Others'], value=[0,1,2], inplace=True)
            df['workex'].replace(to_replace=['No','Yes'], value=[0,1], inplace=True)
            df['specialisation'].replace(to_replace=['Mkt&Fin','Mkt&HR'], value=[0,1], inplace=True)
            df['status'].replace(to_replace=['Placed','Not Placed'],value=[1,0], inplace=True)

            # 4. Splitting the data into X and y
            X, y = self.split_features_labels(df, target_col='status')
           
            # 5. Saving the clean data to a csv
            # df.to_csv('clea')
            # 6. Splitting data into train and test
            X_train, X_test, y_train, y_test = self.split_train_test(X,y)

            # 7. Standardize the data
            logging.info(f"X_train columns-->> {X_train.columns}")
            self.standardize_data(X_train, X_test, "scaling.pkl")

            # 8. Saving the csv file
            self.save_csv(df, dir_name=os.path.join("code","clean_data"), file_name="clean_train.csv")
            logging.info(f"************ Preprocessing Pipeline executed Sucessfully!! ************")
        except Exception as e:
            logging.exception("Exception Occurs -->> {e}")
