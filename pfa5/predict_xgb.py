import pandas as pd
import joblib
import sys
import json

# Load model + encoders + scaler
model = joblib.load("xgb_adherence_model.pkl")
encoders = joblib.load("encoders.pkl")
scaler = joblib.load("scaler.pkl")

def predict_adherence(patient):
    df = pd.DataFrame([patient])

    # Encode categorical columns
    for col, enc in encoders.items():
        df[col] = enc.transform(df[col])

    # Convert date
    df["Treatment Start Date"] = pd.to_datetime(df["Treatment Start Date"])
    df["Start_Year"] = df["Treatment Start Date"].dt.year
    df["Start_Month"] = df["Treatment Start Date"].dt.month
    df["Start_Day"] = df["Treatment Start Date"].dt.day
    df = df.drop("Treatment Start Date", axis=1)

    # Scale numeric
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    df[numeric_cols] = scaler.transform(df[numeric_cols])

    # Predict
    prediction = model.predict(df)[0]
    return round(prediction, 2)


if __name__ == "__main__":
    patient = json.loads(sys.argv[1])
    print(predict_adherence(patient))
