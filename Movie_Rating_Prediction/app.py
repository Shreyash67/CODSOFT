from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load the pre-trained model using pickle
with open("D:\Movie_Rating_Prediction\model.pkl", "rb") as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'GET':
        return render_template("index.html")
    else:
        try:
            # Get input values from the form
            year = int(request.form['Year'])
            duration = float(request.form['Duration'])
            votes = float(request.form['Votes'])

            # Make prediction using the loaded model
            input_data = np.array([[year, duration, votes]])
            prediction = model.predict(input_data)

            return render_template('index.html', prediction=f"Predicted Rating: {prediction[0]:.2f}")
        except Exception as e:
            return render_template('index.html', prediction=f"Error: {e}")

if __name__ == '__main__':
    app.run(debug=True)