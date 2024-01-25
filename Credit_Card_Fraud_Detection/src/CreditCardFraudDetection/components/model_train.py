from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from src.CreditCardFraudDetection.components.data_ingestion import TrainingXY
import pandas as pd
from src.CreditCardFraudDetection.logger import logging
import pickle

class ModelTraining:
    def __init__(self, data_file_path):
        self.data_file_path = data_file_path
        self.models = {
            "Logistic Regression": LogisticRegression(),
            "Random Forest Classifier": RandomForestClassifier(),
        }

    def load_data(self):
        try:
            data = pd.read_csv(self.data_file_path)
            return data
        except FileNotFoundError:
            logging.error(f"File not found: {self.data_file_path}")
            raise
        except Exception as e:
            logging.error(f"An error occurred while loading data: {e}")
            raise

    def data_preprocessing(self):
        # Use the existing train-test split logic
        trainer = TrainingXY(self.data_file_path)
        x_train, x_test, y_train, y_test = trainer.train_test()

        # Train the models
        self.train_models(x_train, y_train)

        # Evaluate the models
        metrics = self.evaluate_models(x_test, y_test)

        # Get the best model
        best_model_name, best_model_score = self.get_best_model(metrics)

        logging.info(f"The best model is '{best_model_name}' with Accuracy: '{best_model_score['Accuracy']}', F1 Score: '{best_model_score['F1 Score']}'")

        # Save the best model as a pickle file
        with open("model.pkl", 'wb') as model_file:
            pickle.dump(self.models[best_model_name], model_file)
        logging.info(f"The best model '{best_model_name}' saved as 'model.pkl'")

    def train_models(self, x_train, y_train):
        for name, model in self.models.items():
            model.fit(x_train, y_train)

    def evaluate_models(self, x_test, y_test):
        logging.info("------------ Model Evaluation Started ------------")
        metrics = {}
        for model_name, model in self.models.items():
            predictions = model.predict(x_test)
            accuracy = accuracy_score(y_test, predictions)
            f1 = f1_score(y_test, predictions)
            metrics[model_name] = {"Accuracy": accuracy, "F1 Score": f1}
            logging.info(f"Model: {model_name}, Accuracy: {accuracy}, F1 Score: {f1}")
        logging.info("------------ Model Evaluation Completed ------------\n")
        return metrics

    def get_best_model(self, metrics):
        best_model_name = max(metrics, key=lambda k: metrics[k]["Accuracy"])
        best_model_score = metrics[best_model_name]
        logging.info(f"Best Model: {best_model_name}, Accuracy: {best_model_score['Accuracy']}, F1 Score: {best_model_score['F1 Score']}")
        return best_model_name, best_model_score

if __name__ == "__main__":
    # Provide the correct file path to the credit card dataset
    data_file_path = r"D:\Credit_Card_Fraud_Detection\notebooks\data\creditcard.csv"
    model_evaluation = ModelTraining(data_file_path)
    model_evaluation.data_preprocessing()
