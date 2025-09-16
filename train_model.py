import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# -----------------------------
# 1. Load Dataset
# -----------------------------
df = pd.read_csv("Crop_recommendation_final.csv")

categorical_cols = ["soil_type", "season", "region", "irrigation", "fertilizer"]
numerical_cols = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

X = df[categorical_cols + numerical_cols]
y = df["label"]

# -----------------------------
# 2. Encode Categorical Columns
# -----------------------------
ord_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
X[categorical_cols] = ord_enc.fit_transform(X[categorical_cols])

# -----------------------------
# 3. Scale Numerical Columns
# -----------------------------
scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

# -----------------------------
# 4. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# 5. Train Model (RandomForest only)
# -----------------------------
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# -----------------------------
# 6. Evaluate
# -----------------------------
y_pred = model.predict(X_test)
print("RandomForest Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# -----------------------------
# 7. Save Model + Preprocessors Together
# -----------------------------
saved_obj = {
    "model": model,
    "scaler": scaler,
    "ordinal_encoder": ord_enc,
    "categorical_cols": categorical_cols,
    "numerical_cols": numerical_cols
}

joblib.dump(saved_obj, "best_crop_model.pkl")
print("✅ Model, scaler, and encoder saved inside best_crop_model.pkl")
