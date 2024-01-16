from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from src.components.training_testing import Training_x_y
import pandas as pd
import pickle

class ModelTraining:

    def __init__(self):
        self.models = {
            "Linear Regression": LinearRegression(),
            "Lasso": Lasso(),
            "Ridge": Ridge(),
            "ElasticNet": ElasticNet(),
            "Random Forest Regression": RandomForestRegressor(),
            "SVR": SVR(),
            "KNN Regression": KNeighborsRegressor(),
        }

    def train(self, x_train, y_train):
        for name, model in self.models.items():
            model.fit(x_train.values.reshape(-1, 1), y_train)

    def predict(self, model_name, x_data):
        model = self.models.get(model_name)
        if model:
            return model.predict(x_data.values.reshape(-1, 1))
        else:
            raise ValueError(f"Model '{model_name}' not found.")

# Load the data
data_file_path = 'D:\\Sale_Prediction\\notebooks\\data\\advertising.csv'
data = pd.read_csv(data_file_path)
obj2 = Training_x_y(data)  # Assuming you have a class named Training_x_y that splits data
x_train, x_test, y_train, y_test = obj2.train_test()

# Train the models
model_obj = ModelTraining()
model_obj.train(x_train, y_train)

# Evaluate the models and choose the best one
r2_scores = {}
for model_name in model_obj.models.keys():
    predictions = model_obj.predict(model_name, x_test)
    r2_scores[model_name] = r2_score(y_test, predictions)

# Choose the best model based on the highest R2 score
best_model_name = max(r2_scores, key=r2_scores.get)
best_model_score = r2_scores[best_model_name]

print(f"The best model is {best_model_name} with R2 score: {best_model_score}")

# Train the best model on the entire dataset
best_model_instance = model_obj.models[best_model_name]
best_model_instance.fit(data[['TV', 'Radio', 'Newspaper']].values, data['Sales'])

# Save the best model as model.pkl
with open("model.pkl", "wb") as file:
    pickle.dump(best_model_instance, file)
