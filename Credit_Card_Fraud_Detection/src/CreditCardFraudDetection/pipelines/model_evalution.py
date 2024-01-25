# model_evaluation.py

import pickle
import pandas as pd
from src.CreditCardFraudDetection.components.model_train import ModelTraining
from src.CreditCardFraudDetection.logger import logging

class ModelEvaluation:
    def __init__(self, data_file_path):
        self.data_file_path = data_file_path
        self.model_trainer = ModelTraining(self.data_file_path)

    def evaluate_models(self):
        self.model_trainer.data_preprocessing()  # Use the data_preprocessing method from ModelTraining

if __name__ == "__main__":
    # Provide the correct file path to the credit card dataset
    data_file_path = r"D:\Credit_Card_Fraud_Detection\notebooks\data\creditcard.csv"
    model_evaluation = ModelEvaluation(data_file_path)
    model_evaluation.evaluate_models()
