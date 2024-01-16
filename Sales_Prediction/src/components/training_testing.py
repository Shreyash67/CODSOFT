import pandas as pd
from sklearn.model_selection import train_test_split

class Spliting_x_y:
    def __init__(self, data):
        self.data = data
    
    def split_x_y(self):
        x = self.data[['TV', 'Radio', 'Newspaper']]
        y = self.data["Sales"] 
        return x, y
    
class Training_x_y(Spliting_x_y):
    def __init__(self, data):
        super().__init__(data)

    def train_test(self):
        x, y = self.split_x_y()
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
        return x_train, x_test, y_train, y_test
