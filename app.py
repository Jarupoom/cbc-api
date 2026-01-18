from flask import Flask, request, jsonify
import pickle
import numpy as np
import os

def explain_cbc(data):
    reasons = []

    if data["MCV"] < 80:
        reasons.append("Low MCV indicates microcytic red blood cells")

    if data["MCH"] < 27:
        reasons.append("Low MCH suggests reduced hemoglobin per cell")

    if data["RBC"] > 5.5:
        reasons.append("Elevated RBC count is commonly seen in thalassemia")

    if data["RDW"] > 15:
        reasons.append("High RDW indicates variation in red blood cell size")

    if not reasons:
        reasons.append("CBC parameters are within normal range")

    return reasons

def validate_cbc(data):
    if not (3.0 <= data["RBC"] <= 8.0):
        return "RBC value is out of valid human range"

    if not (5.0 <= data["Hb"] <= 18.0):
        return "Hemoglobin value is out of valid human range"

    if not (60 <= data["MCV"] <= 120):
        return "MCV value is out of valid human range"

    if not (15 <= data["MCH"] <= 40):
        return "MCH value is out of valid human range"

    if not (25 <= data["MCHC"] <= 38):
        return "MCHC value is out of valid human range"

    if not (10 <= data["RDW"] <= 25):
        return "RDW value is out of valid human range"

    return None  # means valid

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
    error = validate_cbc(data)
    if error:
        return jsonify({
            "error": error,
            "note": "Invalid input values. Please recheck CBC data."
        }), 400

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
        risk = "Uncertain Result"
        recommendation = "Repeat CBC test or seek further laboratory confirmation."
    else:
        if prediction == 1:
            status = "Patient"
            risk = "High Risk"
            recommendation = "Recommend hemoglobin electrophoresis and physician evaluation."
        else:
            status = "Normal"
            risk = "Low Risk"
            recommendation = "No abnormal findings detected in screening."

    pattern = classify_pattern(data)
    explanation = explain_cbc(data)
    
    return jsonify({
        "status": status,
        "risk_level": risk,
        "confidence": f"{confidence}%",
        "pattern_analysis": pattern,
        "explanation": explanation,
        "recommendation": recommendation
    })


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host="0.0.0.0", port=port)
