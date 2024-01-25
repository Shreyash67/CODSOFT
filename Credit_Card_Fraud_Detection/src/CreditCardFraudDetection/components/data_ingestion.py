from src.CreditCardFraudDetection.logger import logging
from sklearn.model_selection import train_test_split
import pandas as pd

class DataIngestion:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None  # Do not read the data in the constructor

    def read_data(self):
        self.data = pd.read_csv(self.data_path)

    def x_y_split(self):
        self.read_data()  # Call the read_data method properly

        # Data transformation
        legit = self.data[self.data["Class"] == 0]
        fraud = self.data[self.data["Class"] == 1]

        legit_sample = legit.sample(n=492)
        new_data = pd.concat((legit_sample, fraud), axis=0)

        x = new_data.drop("Class", axis=1)
        y = new_data["Class"]
        logging.info("Splitting data into x and y")
        return x, y

class TrainingXY(DataIngestion):
    def __init__(self, data_path):
        super().__init__(data_path)

    def train_test(self):
        x, y = self.x_y_split()
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
        logging.info("Splitting the data into train and test\n")
        return x_train, x_test, y_train, y_test