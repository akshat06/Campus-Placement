import pickle
# from data_preprocessing import preprocessing
import os
import logging
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, accuracy_score
import pandas as pd
from sklearn.model_selection import train_test_split

logging_str = "[%(asctime)s: %(levelname)s: %(module)s] %(message)s"
log_dir = "code\logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename= os.path.join(log_dir,"running_logs.log"),level=logging.INFO, format=logging_str, filemode="a")

class ModelTraining:
    def __init__(self) -> None:
        self.train_file_path = os.path.join("code","clean_data","clean_train.csv")
        # self.preprocess = preprocessing.Preprocessing(self.train_file_path)

    def get_data(self):
        try:
            df = pd.read_csv(self.train_file_path)
            logging.info(f"Filename is : {self.train_file_path}")
            logging.info(f"This is Actual Dataframe: {df.head()}")
            logging.info(f"Shape of data is: {df.shape}")
            return df
        except Exception as e:
            return logging.exception(f"Exception-->> {e}")

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

    def standardize_training_data(self, X_train, X_test, filename):
        try:
            logging.info("Standardizing Training Data")
            # self.train_data = self.preprocess.get_data()
            # self.X, self.y = self.preprocess.split_features_labels(df=self.train_data, target_col="status")
            # self.X_train, self.X_test, self.y_train, self.y_test = self.preprocess.split_train_test(self.X, self.y)
            self.scaler = pickle.load(open(filename,"rb"))
            scaled_X_train = self.scaler.fit_transform(X_train)
            scaled_X_test = self.scaler.transform(X_test)
            logging.info("Training data Standardized Sucessfully!!")
            logging.info(f"X_train -->> {scaled_X_train[:5]}")
            logging.info(f"X_test -->> {scaled_X_test[:5]}")
            return scaled_X_train, scaled_X_test

        except Exception as e:
            logging.exception(f"Exception occured in standarizing training data-->> {e}")

    def model_LogisticRegression(self, X, y):
        try:
            logging.info("Training Logistic Regression Model..")
            model_logistic = LogisticRegression()
            model_logistic.fit(X,y)
            return model_logistic
        except Exception as e:
            logging.exception(f"Exception Occured in Training Logistic Regression-->> {e}")

    def model_DecisionTree(self, X,y):
        try:
            logging.info("Training Decision Tree Model..")
            model_dt = DecisionTreeClassifier()
            model_dt.fit(X,y)
            return model_dt
        except Exception as e:
            logging.exception(f"Exception Occured in Training DecisionTree-->> {e}")

    def model_RandomForest(self, X,y):
        try:
            logging.info("Training Random Forest Model..")
            model_rf = RandomForestClassifier()
            model_rf.fit(X,y)
            return model_rf
        except Exception as e:
            logging.exception(f"Exception Occured in Training RandomForest-->> {e}")

    def model_XGBoost(self, X, y):
        try:
            logging.info("Training XGBoost Model..")
            model_xgb = XGBClassifier()
            model_xgb.fit(X,y)
            return model_xgb
        except Exception as e:
            logging.exception(f"Exception Occured in Training XGBoost-->> {e}")

    def performance_evaluation(self, model_dict,X_train,X_test,y_train,y_test):
        try:
            plot_dir = "code\plots"
            os.makedirs(plot_dir, exist_ok=True)
            performance_df = pd.DataFrame()
            performance_dict = dict()
            model_name = []
            train_score = []
            test_score = []
            precision = []
            recall = []
            f1 = []
            accuracy = []
            for name,model in model_dict.items():
                logging.info(f"Training Started for {name}")
                model_name.append(name)
                model.fit(X_train, y_train)
                logging.info(f"Training Score : {model.score(X_train, y_train)}")
                train_score.append(model.score(X_train, y_train))
                logging.info(f"Testing Score: {model.score(X_test, y_test)}")
                test_score.append(model.score(X_test, y_test))
                y_pred = model.predict(X_test)
                logging.info(f"Accuracy Score: {accuracy_score(y_true=y_test, y_pred=y_pred)}")
                accuracy.append(accuracy_score(y_true=y_test, y_pred=y_pred))
                logging.info(f"Precision Score: {precision_score(y_true=y_test, y_pred=y_pred)}")
                precision.append(precision_score(y_true=y_test, y_pred=y_pred))
                logging.info(f"Recall Score: {recall_score(y_true=y_test, y_pred=y_pred)}")
                recall.append(recall_score(y_true=y_test, y_pred=y_pred))
                logging.info(f"F1 Score: {f1_score(y_true=y_test, y_pred=y_pred)}")
                f1.append(f1_score(y_true=y_test, y_pred=y_pred))
                logging.info(f"Confusion Matrix for {name} is as follows:")
                ConfusionMatrixDisplay(confusion_matrix(y_true=y_test, y_pred=y_pred), display_labels=model.classes_).plot()
                # plt.show()
                plt_path = os.path.join(plot_dir,name+".jpg")
                plt.savefig(plt_path)
                logging.info(f"Saved Confusion matrix for {name} at {plt_path}")
                logging.info("*"*50)
                d = {'model_name':model_name, 'Training_score':train_score,'Testing_Score':test_score,'accuracy_score':accuracy,'Precision_Score':precision,
                    'Recall_Score':recall,'F1_Score':f1}
                performance_df = pd.DataFrame(d)
                performance_df.to_csv("model_evaluation.csv", index=False)
        except Exception as e:
            logging.exception("Exception Occured in Performance Evaluation -->> {e}")

    # def determine_best_threshold():
    #     pass

    # def plot_roc_auc_curve():
    #     pass

    def hyperparameter_tuning(self):
        pass
    
    def training_flow(self):
        logging.info("\n\n********************************** Training Pipeline Begins ***************************************")
        # 1. Getting the training data from cleaned training file
        train_data = self.get_data()

        # 2. Splitting features and Labels
        X, y = self.split_features_labels(df=train_data, target_col="status")

        # 3. Splitting data into training and testing
        X_train, X_test, y_train, y_test = self.split_train_test(X, y)

        # 4. Standardizing the data
        file_name = os.path.join("code","model_file","scaling.pkl")
        scaled_X_train, scaled_X_test = self.standardize_training_data(X_train, X_test, file_name)

        # 5. Applying Logistic Regression
        model_log = self.model_LogisticRegression(X_train, y_train)

        # 6. Applying Decision Tree
        model_dt = self.model_DecisionTree(X_train, y_train)

        # 7. Applying Random Forest
        model_rf = self.model_RandomForest(X_train, y_train)

        # 8. Applying XGBoost
        model_xgb = self.model_XGBoost(X_train, y_train)

        # 9. Let's do the Performance Evaluation for all the models
        models = {'logistic':model_log,
         'decision_tree': model_dt,
         'randomforest':model_rf,
         'xgb': model_xgb
         }
        self.performance_evaluation(models ,scaled_X_train, scaled_X_test,y_train,y_test)
