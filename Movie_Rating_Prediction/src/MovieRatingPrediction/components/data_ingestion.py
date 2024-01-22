import pandas as pd
from src.MovieRatingPrediction.logger import logging
from sklearn.model_selection import train_test_split

class DataIngestion:
    def __init__(self, data):
        self.data = data

    def x_y_split(self):
        x = self.data[["Year", "Duration", "Votes"]]    
        y = self.data["Rating"]
        logging.info("Splitting into x and y")
        return x, y

class TrainingXY(DataIngestion):
    def __init__(self, data):
        super().__init__(data)

    def train_test(self):
        x, y = self.x_y_split()
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        logging.info("Splitting the data into train and test\n")
        return x_train, x_test, y_train, y_test
