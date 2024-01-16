from src.components.training_testing import Training_x_y
from src.components.model_fit_pred import ModelTraining
from sklearn.metrics import r2_score
import pandas as pd
import pickle

# Load the data
data_file_path = 'D:\\Sale_Prediction\\notebooks\\data\\advertising.csv'
data = pd.read_csv(data_file_path)
obj2 = Training_x_y(data)  # Pass the data to the constructor
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
best_model_instance.fit(x_train.values.reshape(-1, 1), y_train)

# Save the entire ModelTraining instance as model.pkl
with open("model.pkl", "wb") as file:
    pickle.dump(model_obj, file)
