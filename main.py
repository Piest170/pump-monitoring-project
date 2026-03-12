import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ---------------------------
# LOAD DATASET
# ---------------------------

df = pd.read_csv("sensor_sample.csv").drop(columns=["Unnamed: 0"])

df["timestamp"] = pd.to_datetime(df["timestamp"])

# ตรวจจำนวน NORMAL vs BROKEN
print("Machine Status Distribution:")
print(df["machine_status"].value_counts())

# Sensor columns
sensor_cols = [col for col in df.columns if "sensor" in col]

# ---------------------------
# FEATURE ENGINEERING
# ---------------------------

# Sensor variance
df["sensor_variance"] = df[sensor_cols].std(axis=1)

# Rolling mean
df["rolling_mean"] = df[sensor_cols].mean(axis=1).rolling(window=5).mean()

# Rolling std
df["rolling_std"] = df[sensor_cols].mean(axis=1).rolling(window=5).std()

# Fill missing values
df = df.bfill()

# ---------------------------
# HEALTH SCORE
# ---------------------------

scaler = MinMaxScaler(feature_range=(0, 100))

df["health_score"] = scaler.fit_transform(df[["sensor_variance"]])

df["health_score"] = 100 - df["health_score"]

# ---------------------------
# ANOMALY DETECTION
# ---------------------------

model_anomaly = IsolationForest(
    n_estimators=100,
    contamination=0.01,
    random_state=42
)

model_anomaly.fit(df[sensor_cols])

df["anomaly"] = model_anomaly.predict(df[sensor_cols])

# ---------------------------
# FAILURE LABEL
# ---------------------------

df["failure_label"] = df["machine_status"].apply(
    lambda x: 1 if x == "BROKEN" else 0
)

# ---------------------------
# MACHINE LEARNING MODEL
# ---------------------------

features = sensor_cols + ["sensor_variance", "rolling_mean", "rolling_std"]

X = df[features]

y = df["failure_label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model_rf = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42
)

model_rf.fit(X_train, y_train)

# Evaluate model
y_pred = model_rf.predict(X_test)

print("Model Evaluation:")
print(classification_report(y_test, y_pred))

# Predict failure for all data
df["failure_prediction"] = model_rf.predict(X)

# ---------------------------
# SENSOR IMPORTANCE
# ---------------------------

importances = model_rf.feature_importances_

importance_df = pd.DataFrame({
    "feature": features,
    "importance": importances
})

importance_df = importance_df.sort_values(
    by="importance",
    ascending=False
)

print("\nTop Important Sensors:")
print(importance_df.head(10))

# Save importance file
importance_df.to_csv("sensor_importance.csv", index=False)

# ---------------------------
# SAVE FINAL DATASET
# ---------------------------

df.to_csv("processed_sensor.csv", index=False)

print("Processing Complete")