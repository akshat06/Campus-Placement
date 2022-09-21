import pickle
import os
import logging
import pickle

logging_str = "[%(asctime)s: %(levelname)s: %(module)s] %(message)s"
log_dir = "code\logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename= os.path.join(log_dir,"running_logs.log"),level=logging.INFO, format=logging_str, filemode="a")

class Predicition:
    def __init__(self, model_file, data) -> None:
        self.model_file = model_file
        self.data = data

    def load_model(self, model_file):
        logging.info(f"Loading model file {model_file}")
        model = pickle.load(open(model_file,"rb"))
        logging.info(f"Model file {model_file} loaded sucessfully!!")
        return model

    def predict_status(self, model, features):
        logging.info(f"Prediction started for features: {features}")
        prediction = model.predict([features])
        logging.info(f"Prediction for above features is: {prediction[0]}")
        return prediction

    def prediction_flow(self):
        # 1. Get data from WebApp
        # 2. Pass the data into the model
        # 3. Predict the output from the model
        # 4. Return the Status to the WebApp
        logging.info("**************** Prediction Pipeline Started!! ****************")
        model_file = os.path.join("code","model_file","best_model.pkl")
        pred_model = self.load_model(model_file=self.model_file)
        pred_output = self.predict_status(pred_model, self.data)
        logging.info("**************** Prediction Pipeline Ended!! ****************")
        return pred_output

