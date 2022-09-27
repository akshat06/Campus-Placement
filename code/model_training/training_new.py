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
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

logging_str = "[%(asctime)s: %(levelname)s: %(module)s] %(message)s"
log_dir = "code\logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename= os.path.join(log_dir,"running_logs.log"),level=logging.INFO, format=logging_str, filemode="a")

class ModelTraining:
    def __init__(self) -> None:
        self.train_file_path = os.path.join("code","clean_data","preprocessed_data.csv")
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
            logging.info(f"X :\n{X.head()}\n y: {y.head()}")
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
                performance_df.to_csv("model_evaluation_new.csv", index=False)
        except Exception as e:
            logging.exception("Exception Occured in Performance Evaluation -->> {e}")

    def find_best_model(self, evaluation_file, scoring='Precision_Score'):
        try:
            evaluation_df = pd.read_csv(evaluation_file)
            if scoring == 'Precision_Score':
                evaluation_df = evaluation_df.sort_values(by='Precision_Score', ascending=False)
                logging.info(f"Best Model name: {evaluation_df.iloc[0][0]}")
                return evaluation_df.iloc[0][0]
            elif scoring == 'Recall_Score':
                evaluation_df = evaluation_df.sort_values(by='Recall_Score', ascending=False)
                logging.info(f"Best Model name: {evaluation_df.iloc[0][0]}")
                return evaluation_df.iloc[0][0]
            elif scoring == 'accuracy_score':
                evaluation_df = evaluation_df.sort_values(by='accuracy_score', ascending=False)
                logging.info(f"Best Model name: {evaluation_df.iloc[0][0]}")
                return evaluation_df.iloc[0][0]
            elif scoring == 'F1_Score':
                evaluation_df = evaluation_df.sort_values(by='F1_Score', ascending=False)
                logging.info(f"Best Model name: {evaluation_df.iloc[0][0]}")
                return evaluation_df.iloc[0][0]
            else:
                logging.info(f"Got Invalid Scoring Parameter: {scoring}")
        except Exception as e:
            logging.exception("Exception occured at finding best model: {e}")

    def training_best_model(self, params_dict, model_name,X_train,y_train):
        try:
            logging.info(f"Training Best Model for : {model_name}")
            if model_name == 'decision_tree':
                try:
                    model_dt = DecisionTreeClassifier()
                    # grid_search = GridSearchCV(estimator=model_dt, 
                    #         param_grid=params_dict, 
                    #         cv=4, n_jobs=-1, verbose=3, scoring = "precision")
                    random_search = RandomizedSearchCV(estimator=model_dt, 
                            param_distributions=params_dict, 
                            cv=4, n_jobs=-1, verbose=3, scoring = "precision")
                    random_search.fit(X_train, y_train)
                    model = random_search.best_estimator_
                    return model
                except Exception as e:
                    logging.exception(f"Exception occurs in training Decision Tree: {e}")
                
            elif model_name == 'xgb':
                try:
                    model_xgb = XGBClassifier()
                    random_search = RandomizedSearchCV(estimator=model_xgb, 
                            param_distributions=params_dict, 
                            cv=4, n_jobs=-1, verbose=3, scoring = "precision")
                    random_search.fit(X_train, y_train)
                    model = random_search.best_estimator_
                    return model
                except Exception as e:
                    logging.exception(f"Exception occurs in training XGBoost: {e}")

            elif model_name == 'randomforest':
                try:
                    model_rf = RandomForestClassifier()
                    random_search = RandomizedSearchCV(estimator=model_rf, 
                            param_distributions=params_dict, 
                            cv=4, n_jobs=-1, verbose=3, scoring = "precision")
                    random_search.fit(X_train, y_train)
                    model = random_search.best_estimator_
                    return model
                except Exception as e:
                    logging.exception(f"Exception occurs in training RandomForest: {e}")

            elif model_name == 'logistic':
                try:
                    model_log = LogisticRegression()
                    random_search = RandomizedSearchCV(estimator=model_log, 
                            param_distributions=params_dict, 
                            cv=4, n_jobs=-1, verbose=3, scoring = "precision")
                    random_search.fit(X_train, y_train)
                    model = random_search.best_estimator_
                    return model
                except Exception as e:
                    logging.exception(f"Exception occurs in training Logistic Regression: {e}")

            else:
                logging.error(f"No relevant model found..model is : {model_name}")
        except Exception as e:
            logging.exception(f"Exception occurs in model_training: {e}")

    def model_saving(self, model, model_name, model_dir):
        try:
            logging.info(f"Saving the best model and best model is : {model_name}")
            model_dir = os.path.join(model_dir, model_name)
            pickle.dump(model, open(model_dir,'wb'))
            logging.info(f"Model Saved Sucessfully at : {model_dir}")
        except Exception as e:
            logging.exception(f"Exception occured in saving the model: {e}")
    
    def training_flow(self):
        logging.info("\n\n********************************** Training Pipeline Begins ***************************************")
        # 1. Getting the training data from cleaned training file
        train_data = self.get_data()

        # 2. Splitting features and Labels
        X, y = self.split_features_labels(df=train_data, target_col="status")

        # 3. Standardizing the data
        """Since Bagging and Boosting algorithms do not require Scaling
           so we are commenting the following section of code
        """
        # file_name = os.path.join("code","model_file","scaling.pkl")
        # scaled_data = self.standardize_training_data(file_name)

        # 3. Splitting data into training and testing
        # X = scaled_data.drop(columns=['status'], axis=1)
        # y = scaled_data['status']
        X_train, X_test, y_train, y_test = self.split_train_test(X, y)

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
        #  'xgb': model_xgb
         }
        self.performance_evaluation(models ,X_train, X_test, y_train,y_test)

        # 10. Finding the best model among the above ones
        evaluation_file = "model_evaluation_new.csv"
        logging.info(f"Evaluation File dir is: {evaluation_file}")
        best_model_name = self.find_best_model(evaluation_file, scoring='Precision_Score')
        logging.info(f" ******* Best Model is : {best_model_name} ******** ")

        # 11. Training the best model
        # best_model_name = 'randomforest'
        if best_model_name == 'xgb':
            params_dict = {'gamma': [0,0.1,0.2,0.4,0.8,1.6,3.2,6.4,12.8,25.6,51.2,102.4, 200],
              'learning_rate': [0.01, 0.03, 0.06, 0.1, 0.15, 0.2, 0.25, 0.300000012, 0.4, 0.5, 0.6, 0.7],
              'max_depth': [5,6,7,8,9,10,11,12,13,14],
              'n_estimators': [50,65,80,100,115,130,150],
              'reg_alpha': [0,0.1,0.2,0.4,0.8,1.6,3.2,6.4,12.8,25.6,51.2,102.4,200],
              'reg_lambda': [0,0.1,0.2,0.4,0.8,1.6,3.2,6.4,12.8,25.6,51.2,102.4,200]}
            best_model = self.training_best_model(params_dict, best_model_name, X_train,y_train)
            
        elif best_model_name == 'decision_tree':
            params_dict = {
                                'max_depth': [2, 3, 5, 10, 20],
                                'min_samples_leaf': [5, 10, 20, 50, 100],
                                'criterion': ["gini", "entropy"]
                            }
            best_model = self.training_best_model(params_dict, best_model_name, X_train,y_train)

        elif best_model_name == 'randomforest':
            # Number of trees in random forest
            n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
            # Number of features to consider at every split
            max_features = ['auto', 'sqrt']
            # Maximum number of levels in tree
            max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
            max_depth.append(None)
            # Minimum number of samples required to split a node
            min_samples_split = [2, 5, 10]
            # Minimum number of samples required at each leaf node
            min_samples_leaf = [1, 2, 4]
            # Method of selecting samples for training each tree
            bootstrap = [True, False]

            params_dict = {'n_estimators': n_estimators,
                        'max_features': max_features,
                        'max_depth': max_depth,
                        'min_samples_split': min_samples_split,
                        'min_samples_leaf': min_samples_leaf,
                        'bootstrap': bootstrap}
            best_model = self.training_best_model(params_dict, best_model_name, X_train,y_train)

        else:
            solvers = ['newton-cg', 'lbfgs', 'liblinear']
            penalty = ['l2']
            c_values = [100, 10, 1.0, 0.1, 0.01]
            # define grid search
            params_dict = dict(solver=solvers,penalty=penalty,C=c_values)
            best_model = self.training_best_model(params_dict, best_model_name, X_train,y_train)
        # 12. Saving the best model
        model_name = "best_model.pkl"
        model_dir = os.path.join("code","model_file")
        self.model_saving(best_model, model_name, model_dir)
        return best_model_name
        logging.info("\n\n********************************** Training Pipeline Ends ***************************************")