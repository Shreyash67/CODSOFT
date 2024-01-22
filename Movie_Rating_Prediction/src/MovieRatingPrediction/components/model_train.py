from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
from src.MovieRatingPrediction.components.data_transformation import DataTransform
from src.MovieRatingPrediction.components.data_ingestion import TrainingXY
import pandas as pd
from src.MovieRatingPrediction.logger import logging
import pickle

class ModelTraining:
    def __init__(self, data_file_path):
        self.data_file_path = data_file_path
        self.models = {
        "Linear Regression": LinearRegression(),
        "Lasso": Lasso(),
        "Ridge": Ridge(),
        "ElasticNet": ElasticNet(),
        "Random Forest Regression": RandomForestRegressor(),
        "Gradient Boosting Regression": GradientBoostingRegressor(),
        "SVR": SVR(),
        "KNN Regression": KNeighborsRegressor(),
        "Decision Tree Regression": DecisionTreeRegressor(),
        "AdaBoost Regression": AdaBoostRegressor(),
    }

    def load_data(self):
        try:
            data = pd.read_csv(self.data_file_path, encoding="windows-1252")
            return data
        except FileNotFoundError:
            logging.error(f"File not found: {self.data_file_path}")
            raise
        except Exception as e:
            logging.error(f"An error occurred while loading data: {e}")
            raise

    def data_preprocessing(self, data):
        transformer = DataTransform(data)
        transformed_data = transformer.data_transform()
        return transformed_data

    def train_models(self, x_train, y_train):
        for name, model in self.models.items():
            model.fit(x_train, y_train)

    def evaluate_models(self, x_test, y_test):
        logging.info("------------ Model Evaluation Started ------------")
        metrics = {}
        for model_name, model in self.models.items():
            predictions = model.predict(x_test)
            mae = mean_absolute_error(y_test, predictions)
            metrics[model_name] = mae
            logging.info(f"Model: {model_name}, MAE: {mae}")
        logging.info("------------ Model Evaluation Completed ------------\n")
        return metrics


    def get_best_model(self, metrics):
        best_model_name = min(metrics, key=metrics.get)
        best_model_score = metrics[best_model_name]
        logging.info(f"Best Model: {best_model_name}, MAE: {best_model_score}")
        return best_model_name, best_model_score


    def train_evaluate_best_model(self):
        # Load data and perform preprocessing
        data = self.load_data()
        transformed_data = self.data_preprocessing(data)

        # Use the existing train-test split logic
        trainer = TrainingXY(transformed_data)
        x_train, x_test, y_train, y_test = trainer.train_test()

        # Train the models
        self.train_models(x_train, y_train)

        # Evaluate the models
        r2_scores = self.evaluate_models(x_test, y_test)

        # Get the best model
        best_model_name, best_model_score = self.get_best_model(r2_scores)

        logging.info(f"The best model is '{best_model_name}' with MAE score: '{best_model_score}'")

        # Save the best model as a pickle file
        with open("model.pkl", 'wb') as model_file:
            pickle.dump(self.models[best_model_name], model_file)
        logging.info(f"The best model '{best_model_name}' saved as 'model.pkl'")