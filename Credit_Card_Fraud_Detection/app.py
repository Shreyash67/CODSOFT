# app.py

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

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'GET':
        return render_template("index.html")
    else:
        try:
            # Get input values from the form
            time = float(request.form['Time'])
            v1 = float(request.form['V1'])
            v2 = float(request.form['V2'])
            v3 = float(request.form['V3'])
            v4 = float(request.form['V4'])
            v5 = float(request.form['V5'])
            v6 = float(request.form['V6'])
            v7 = float(request.form['V7'])
            v8 = float(request.form['V8'])
            v9 = float(request.form['V9'])
            v10 = float(request.form['V10'])
            v11 = float(request.form['V11'])
            v12 = float(request.form['V12'])
            v13 = float(request.form['V13'])
            v14 = float(request.form['V14'])
            v15 = float(request.form['V15'])
            v16 = float(request.form['V16'])
            v17 = float(request.form['V17'])
            v18 = float(request.form['V18'])
            v19 = float(request.form['V19'])
            v20 = float(request.form['V20'])
            v21 = float(request.form['V21'])
            v22 = float(request.form['V22'])
            v23 = float(request.form['V23'])
            v24 = float(request.form['V24'])
            v25 = float(request.form['V25'])
            v26 = float(request.form['V26'])
            v27 = float(request.form['V27'])
            v28 = float(request.form['V28'])
            amount = float(request.form['Amount'])

            # Make prediction using the loaded model
            input_data = np.array([[time, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, 
                                    v14, v15, v16, v17,v18, v19, v20, v21, v22, v23, 
                                    v24, v25, v26, v27, v28, amount]])
            prediction = model.predict(input_data)

            fraud_status = "Fraudulent" if prediction[0] == 1 else "Not Fraudulent"

            return render_template('index.html', prediction=f"The transaction is predicted as {fraud_status}.")
        except Exception as e:
            return render_template('index.html', prediction=f"Error: {e}")


if __name__ == '__main__':
    app.run(debug=True)
