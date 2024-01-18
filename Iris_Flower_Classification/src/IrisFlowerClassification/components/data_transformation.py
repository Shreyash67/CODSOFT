from src.IrisFlowerClassification.logger import logging
from sklearn.preprocessing import LabelEncoder

class data_transform:
    def __init__(self,data):
        self.data=data

    def data_encoding(self):
        le = LabelEncoder()
        self.data["species"] = le.fit_transform(self.data["species"])
        logging.info("Initializing Data Preprocessing")
        return self.data
