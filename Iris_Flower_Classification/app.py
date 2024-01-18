from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load the pre-trained model using pickle
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template("index.html")
    else:
        try:
            # Get input values from the form
            sepal_length = float(request.form['sepal_length'])
            sepal_width = float(request.form['sepal_width'])
            petal_length = float(request.form['petal_length'])
            petal_width = float(request.form['petal_width'])

            # Ensure that the input values are numerical
            if any(np.isnan([sepal_length, sepal_width, petal_length, petal_width])):
                raise ValueError("Invalid input. Please enter numerical values for all features.")

            # Make prediction using the pre-trained model
            input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
            prediction = int(model.predict(input_data)[0])  # Convert to integer

            # Map numerical prediction to corresponding species
            species_mapping = {0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'}
            
            # Handle cases where the prediction value is not in species_mapping
            predicted_species = species_mapping.get(prediction, 'Unknown Species')

            return render_template("index.html", prediction=f"species: {predicted_species}")
        except ValueError as e:
            return render_template("index.html", error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
