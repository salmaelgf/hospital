import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from xgboost import XGBRegressor
import joblib

print("📥 Loading dataset...")

df = pd.read_csv("synthetic_mental_health_5000.csv")
print("Dataset loaded!")
print(df.head(), "\n")

# ============================================================
# 1. CLEANING
# ============================================================
print("🧹 Cleaning data...")

df = df.drop_duplicates()
df = df.dropna()

# Convert dates
df["Treatment Start Date"] = pd.to_datetime(df["Treatment Start Date"])
df["Start_Year"] = df["Treatment Start Date"].dt.year
df["Start_Month"] = df["Treatment Start Date"].dt.month
df["Start_Day"] = df["Treatment Start Date"].dt.day
df = df.drop("Treatment Start Date", axis=1)

print("Cleaning done.\n")

# ============================================================
# 2. ENCODING CATEGORICAL VARIABLES
# ============================================================
print("🔄 Encoding categorical variables...")

label_cols = [
    "Gender",
    "Diagnosis",
    "Medication",
    "Therapy Type",
    "Outcome",
    "AI-Detected Emotional State"
]

encoders = {}
for col in label_cols:
    enc = LabelEncoder()
    df[col] = enc.fit_transform(df[col])
    encoders[col] = enc

print("Encoding done.\n")

# ============================================================
# 3. TRAIN/TEST SPLIT
# ============================================================
print("✂ Splitting dataset...")

X = df.drop("Adherence to Treatment (%)", axis=1)
y = df["Adherence to Treatment (%)"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training samples : {len(X_train)}")
print(f"Testing samples  : {len(X_test)}\n")

# ============================================================
# 4. SCALING
# ============================================================
print("📏 Scaling numeric features...")

scaler = StandardScaler()
num_cols = X_train.select_dtypes(include=["float64", "int64"]).columns

X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

print("Scaling done.\n")

# ============================================================
# 5. TRAINING XGBOOST
# ============================================================
print("⚡ Training XGBoost model...")

model = XGBRegressor(
    objective="reg:squarederror",
    n_estimators=400,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)

print("Model training complete!\n")

# ============================================================
# 6. EVALUATION
# ============================================================
print("📊 Evaluating model...")

preds = model.predict(X_test)

mae = mean_absolute_error(y_test, preds)
rmse = mean_squared_error(y_test, preds) ** 0.5
r2 = r2_score(y_test, preds)

print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.4f}\n")

# ============================================================
# 7. FEATURE IMPORTANCE
# ============================================================
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("🔥 Top 10 most important features:")
print(importance.head(10), "\n")

importance.to_csv("feature_importance.csv", index=False)

# ============================================================
# 8. SAVE MODEL
# ============================================================
print("💾 Saving model & preprocessors...")

joblib.dump(model, "xgb_adherence_model.pkl")
joblib.dump(encoders, "encoders.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\n🎉 Model saved successfully!")
print("🚀 Training pipeline complete!")
