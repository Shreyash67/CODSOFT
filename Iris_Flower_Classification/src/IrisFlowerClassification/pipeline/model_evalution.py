import pickle
from src.IrisFlowerClassification.components.model_train import ModelTraining
from src.IrisFlowerClassification.logger import logging
from src.IrisFlowerClassification.components.data_ingestion import Training_x_y
import pandas as pd

class Model_Evaluation:
    def __init__(self, data_file_path):
        self.data_file_path = data_file_path

    def evaluate_models(self):
        # Creating an instance of ModelTraining
        model_trainer = ModelTraining(self.data_file_path)

        # Load data and perform preprocessing
        data = model_trainer.load_data()
        transformed_data = model_trainer.data_preprocessing(data)

        # Split the data into training and testing sets
        obj2 = Training_x_y(transformed_data)
        x_train, x_test, y_train, y_test = obj2.train_test()

        # Ensure x_test has the same columns as x_train
        x_test = x_test[x_train.columns]

        # Train the models
        model_trainer.train_models(x_train, y_train)
        logging.info("------------ Models Training Completed ------------")

        # Evaluate the models
        r2_scores = model_trainer.evaluate_models(x_test, y_test)

        # Get the best model
        best_model_name, best_model_score = model_trainer.get_best_model(r2_scores)

        logging.info(f"The best model is '{best_model_name}' with R2 score: '{best_model_score}'")

        # Save the best model as a pickle file
        with open("model.pkl", 'wb') as model_file:
            pickle.dump(model_trainer.models[best_model_name], model_file)
        logging.info(f"The best model '{best_model_name}' saved as 'model.pkl'") 
  

if __name__ == "__main__":
    data_file_path = r'D:\Iris_Flower_Classification\notebooks\data\IRIS.csv'
    model_evaluation = Model_Evaluation(data_file_path)
    model_evaluation.evaluate_models()