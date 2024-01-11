import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('D:\\Titanic_Survival_Prediction\\notebook\\model.pkl', 'rb'))

@app.route('/')
def home_page():
    return render_template("index.html")

@app.route("/predict", methods=["GET","POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("result.html")
    else:
        Pclass = float(request.form.get('Pclass'))
        Sex = float(request.form.get('Sex'))

        
        input_data = np.array([[Pclass, Sex]])
        

        # Make prediction using the loaded model
        prediction = model.predict(input_data)

        print("prediction value: ", prediction)

        result = ""
        if prediction == 1:
            result = "passenger survived"
        else:
            result = "passenger not survived"

        return render_template('result.html', prediction_text=result)

# Execution begins
if __name__ == '__main__':
    app.run(port=8080)
