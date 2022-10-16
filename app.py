# from data_preprocessing.preprocessing import Preprocessing
from data_preprocessing.preprocessing_new import Preprocessing
# from model_training.training import ModelTraining
from model_training.training_new import ModelTraining
from model_prediction.prediction import Predicition
import os
import pandas as pd
import logging
import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd

logging_str = "[%(asctime)s: %(levelname)s: %(module)s] %(message)s"
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename= os.path.join(log_dir,"running_logs.log"),level=logging.INFO, format=logging_str, filemode="a")

# model = pickle.load(open('model_file\\best_model.pkl','rb'))
model = pickle.load(open('best_model.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    try:
        if request.method == "POST" :
            # gender, ssc_p, hsc_p, degree_p, workex, etest_p, specialization, mba_p, hsc_s, degree_t
        
            Gender= str(request.form['gender'])
            Ssc_p= float(request.form['ssc_p'])
            # Ssc_b = str(request.form['ssc_b'])
            Hsc_p= float(request.form['hsc_p'])
            # Hsc_b = str(request.form['hsc_b'])
            Hsc_s = str(request.form['hsc_s'])
            Degree_p= float(request.form['degree_p'])
            Degree_t = str(request.form['degree_t'])
            Workex = str(request.form['workex'])
            Etest_p = float(request.form['etest_p'])
            Specialization = str(request.form['specialization'])
            MBA_p = float(request.form['mba_p'])
            # salary = float(request.form['salary'])
            
            print(f"Gender: {Gender}")
            print(f"Ssc_p:{Ssc_p}")
            # print(f"Ssc_b:{Ssc_b}")
            print(f"Hsc_p:{Hsc_p}")
            # print(f"Hsc_b:{Hsc_b}")
            print(f"Hsc_s:{Hsc_s}")
            print(f"Degree_p:{Degree_p}")
            print(f"Degree_t:{Degree_t}")
            print(f"Workex:{Workex}")
            print(f"Etest:{Etest_p}")
            print(f"Specialization:{Specialization}")
            print(f"MBA_p:{MBA_p}")
            # print(f"salary:{salary}")
            pred_list = []
            if Gender=='Female':
                Gender=1
                pred_list.append(Gender)
            else: 
                Gender=0
                pred_list.append(Gender)
            
            pred_list.append(Ssc_p)
            pred_list.append(Hsc_p)
            pred_list.append(Degree_p)

            if Workex == 'Yes':
                pred_list.append(1)
            else:
                pred_list.append(0)

            pred_list.append(Etest_p)


            if Specialization == "Mkt&Fin":
                pred_list.append(0)
            else:
                pred_list.append(1)
            
            pred_list.append(MBA_p)

            if Hsc_s == 'Arts':
                pred_list.append(1)
                pred_list.append(0)
                pred_list.append(0)

            elif Hsc_s == 'Commerce':
                pred_list.append(0)
                pred_list.append(1)
                pred_list.append(0)

            else:
                pred_list.append(0)
                pred_list.append(0)
                pred_list.append(1)
            
            if Degree_t == 'Comm&Mgmt':
                pred_list.append(1)
                pred_list.append(0)
                pred_list.append(0)
            
            elif Degree_t == 'Others':
                pred_list.append(0)
                pred_list.append(1)
                pred_list.append(0)
            
            else:
                pred_list.append(0)
                pred_list.append(0)
                pred_list.append(1)
            

            
            # Pred_args=[Gender, Ssc_p, Ssc_b, Hsc_p, Hsc_b, Hsc_s, Degree_p, Degree_t, Workex,Etest_p,Specialization,MBA_p, salary]
            logging.info(f"Prediction List: {pred_list}")
            pred_args=np.array(pred_list)
            pred_args=pred_args.reshape(1,-1)
            
            y_pred=model.predict(pred_args)
            logging.info(f"pred: {y_pred}")
            y_pred=y_pred[0]
            logging.info(f"Y_pred:-->> {y_pred}")
            if y_pred == 0:
                image_path = os.path.join("code","templates","work_hard.jpg")
                logging.info("Work Hard!!! Chances are less")
                return render_template('predict.html',prediction="Work Hard!!! Chances are less", user_image=image_path) 
            else:
                image_path = os.path.join("code","templates","celebrate.jpg")
                logging.info("You are Doing well!! You Will Get placements")
                return render_template('predict.html',prediction=" You are Doing well!! You Will Get placements", user_image = image_path) 
    except Exception as e:
        logging.exception(f"Exception occured during Prediction:\n {e}")

# """Data Preprocessing
# """
# train_file_path = os.path.join("dataset","train.csv")
# print(train_file_path)
# pre_process = Preprocessing(filename=train_file_path)
# pre_process.preprocessing_flow()

# """Model Training
# """
# model_train = ModelTraining()
# model_train.training_flow()


# """Model Prediction
# """
# model = os.path.join("model_file","best_model.pkl")
# pred_data = [0,56.0,1,52.0,1,1,52.0,1,0,66.0,1,59.43,288655.4054054054]
# model_pred = Predicition(model_file=model, data=pred_data)
# if model_pred == 0:
#     logging.info("\n\nPrediction-->> Student will not be placed!")
# else:
#     logging.info("\n\nPrediction-->> Student will get placed!!")

@app.route('/train',methods=['GET','POST'])
def train():
    try:
        print("______ Inside Train Function________")
        logging.info("______ Inside Train Function________")
        if request.method == "POST" :
            logging.info("Model training Started!!")
            """Data Preprocessing
            """
            train_file_path = os.path.join("code","dataset","train.csv")
            print(train_file_path)
            pre_process = Preprocessing(filename=train_file_path)
            pre_process.preprocessing_flow()
            """Model Training
            """
            model_train = ModelTraining()
            best_model_name = model_train.training_flow()
            return render_template('train.html',best_model=f"Best Model is {best_model_name}")
    except Exception as e:
        logging.exception(f"Exception occured: {e}")
    
if __name__ == "__main__":
    app.run(debug=True)