from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib

app = Flask(__name__)
CORS(app)

# Load model + encoders + scaler
model = joblib.load("xgb_adherence_model.pkl")
encoders = joblib.load("encoders.pkl")
scaler = joblib.load("scaler.pkl")

# ➤ EXACT order of features used during training
FEATURE_ORDER = [
    "Age",
    "Gender",
    "Diagnosis",
    "Symptom Severity (1-10)",
    "Mood Score (1-10)",
    "Sleep Quality (1-10)",
    "Physical Activity (hrs/week)",
    "Medication",
    "Therapy Type",
    "Treatment Duration (weeks)",
    "Stress Level (1-10)",
    "Outcome",
    "Treatment Progress (1-10)",
    "AI-Detected Emotional State",
    "Start_Year",
    "Start_Month",
    "Start_Day"
]

@app.route("/", methods=["GET"])
def home():
    return {"message": "MindMeter XGBoost Adherence API is running 🚀"}

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        # ------------------------------
        # 🔥 Normalize Gender
        # ------------------------------
        if "Gender" in data:
            g = data["Gender"].strip().lower()
            if g in ["femme", "female", "f"]:
                data["Gender"] = "Female"
            elif g in ["homme", "male", "m", "h"]:
                data["Gender"] = "Male"
            else:
                data["Gender"] = "Female"

        # ------------------------------
        # 🔥 Prepare dataframe
        # ------------------------------
        df = pd.DataFrame([data])

        # ------------------------------
        # 🔥 Encode categorical values
        # ------------------------------
        for col, enc in encoders.items():
            df[col] = enc.transform(df[col])

        # ------------------------------
        # 🔥 Convert date to features
        # ------------------------------
        df["Treatment Start Date"] = pd.to_datetime(df["Treatment Start Date"])
        df["Start_Year"] = df["Treatment Start Date"].dt.year
        df["Start_Month"] = df["Treatment Start Date"].dt.month
        df["Start_Day"] = df["Treatment Start Date"].dt.day
        df = df.drop("Treatment Start Date", axis=1)

        # ------------------------------
        # 🔥 Reorder exactly like training
        # ------------------------------
        df = df[FEATURE_ORDER]

        # ------------------------------
        # 🔥 Scale numeric columns
        # ------------------------------
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
        df[numeric_cols] = scaler.transform(df[numeric_cols])

        # ------------------------------
        # 🔥 Prediction
        # ------------------------------
        pred = float(model.predict(df)[0])

        return jsonify({
            "adherence_prediction": round(pred, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
