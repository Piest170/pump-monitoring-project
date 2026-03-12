import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("🏭 Smart Factory Pump Monitoring")

df = pd.read_csv("processed_sensor.csv")
importance = pd.read_csv("sensor_importance.csv")

df["timestamp"] = pd.to_datetime(df["timestamp"])

sensor_cols = [col for col in df.columns if "sensor" in col]

failures = df[df["machine_status"] == "BROKEN"]
anomalies = df[df["anomaly"] == -1]

st.subheader("📋Dataset Overview")
st.write(df.head())

sensor = st.selectbox(
    "Select Sensor",
    sensor_cols
)

latest_health = int(df["health_score"].iloc[-1])

st.metric(
    label="Machine Health Score",
    value=f"{latest_health}%"
)

latest_prediction = df["failure_prediction"].iloc[-1]

if latest_prediction == 1:
    st.error("⚠ HIGH FAILURE RISK")
else:
    st.success("✅ MACHINE NORMAL")

st.subheader(f"🎯{sensor} Trend")

plot_df = df.tail(5000)

fig, ax = plt.subplots(figsize=(12,4))

ax.plot(plot_df["timestamp"], plot_df[sensor], label="Sensor")

ax.scatter(
    anomalies["timestamp"],
    anomalies[sensor],
    color="orange",
    label="Anomaly"
)

ax.scatter(
    failures["timestamp"],
    failures[sensor],
    color="red",
    label="Failure"
)

ax.legend()

st.pyplot(fig)

st.subheader("🔝Top Important Sensors")

top_sensors = importance.head(10)

st.bar_chart(
    top_sensors.set_index("feature")["importance"]
)