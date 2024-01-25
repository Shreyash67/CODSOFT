import pandas as pd

class DataTransform:
    def __init__(self, data_path):
        self.data_path = data_path

    def data_transform(self):
        data = pd.read_csv(self.data_path)  # Load the data from the file path
        legit = data[data["Class"] == 0]
        fraud = data[data["Class"] == 1]

        legit_sample = legit.sample(n=492)
        new_data = pd.concat((legit_sample, fraud), axis=0)

        return new_data    