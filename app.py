from flask import Flask, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load model
model = pickle.load(open("Thalacheck.pkcls", "rb"))

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    # CBC inputs (must match Orange order)
    features = np.array([[
        data["RBC"],
        data["Hb"],
        data["MCV"],
        data["MCH"],
        data["MCHC"],
        data["RDW"]
    ]])

    prediction = model.predict(features)[0]

    # Change meaning, NOT the model

    if prediction == 1:

        status = "Patient"

        risk = "High Risk"

    else:

        status = "Normal"

        risk = "Low Risk"

    return jsonify({

        "status": status,

        "risk_level": risk,
    })

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host="0.0.0.0", port=port)
