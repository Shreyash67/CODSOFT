import pickle
import pandas as pd
from src.MovieRatingPrediction.components.model_train import ModelTraining
from src.MovieRatingPrediction.logger import logging

class ModelEvaluation:
    def __init__(self, data_file_path):
        self.data_file_path = data_file_path
        self.model_trainer = ModelTraining(self.data_file_path)

    def evaluate_models(self):
        # Call the existing train_evaluate_best_model method
        self.model_trainer.train_evaluate_best_model()

if __name__ == "__main__":
    # Provide the file path instead of the DataFrame
    data_file_path = r"D:\Movie_Rating_Prediction\notebooks\data\IMDb Movies India.csv"
    model_evaluation = ModelEvaluation(data_file_path)
    model_evaluation.evaluate_models()
