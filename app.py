from flask import Flask, request, jsonify # type: ignore
import pickle
import numpy as np # type: ignore
import os

def classify_pattern(data):
    # Thalassemia pattern:
    if (
        data["MCV"] < 75 and
        data["MCH"] < 27 and
        data["RBC"] > 5.5 and
        data["RDW"] <= 15
    ):
        return "Likely Thalassemia Trait"

    # Iron Deficiency pattern:
    if (
        data["MCV"] < 80 and
        data["MCH"] < 27 and
        data["RDW"] > 15
    ):
        return "Likely Iron Deficiency Anemia"

    return "No specific anemia pattern detected"

#load Model
app = Flask(__name__)
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

    probabilities = model.predict_proba(features)[0]
    confidence = round(max(probabilities) * 100, 2)

    if confidence < 65:
        status = "Uncertain"
        risk = "Uncertain Result ❌"
        recommendation = "Repeat CBC test or seek further laboratory confirmation."
    else:
        if prediction == 1:
            status = "Patient"
            risk = "High Risk ⚠️"
            recommendation = "Recommend hemoglobin electrophoresis and physician evaluation."
        else:
            status = "Normal"
            risk = "Low Risk ✅"
            recommendation = "No abnormal findings detected in screening."

    pattern = classify_pattern(data)

    return jsonify({
        "Confidence": f"{confidence}%",
        "Status": status,
        "Risk_level": risk,
        "Pattern_analysis": pattern,
        "Guidance": recommendation
    })


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host="0.0.0.0", port=port)
