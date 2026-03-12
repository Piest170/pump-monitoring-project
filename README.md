# Smart Factory Pump Monitoring

## 📌 Project Overview

This project demonstrates a **Predictive Maintenance System** for industrial pumps using sensor data.
The system analyzes multiple sensor readings to monitor machine health, detect anomalies, and predict potential pump failures.

The goal is to simulate a **smart factory monitoring system** where engineers can observe sensor behavior, detect abnormal patterns, and take preventive maintenance actions before machine breakdown occurs.

Key capabilities include:

* Machine Health Score calculation
* Sensor anomaly detection
* Failure prediction using machine learning
* Visualization dashboard for monitoring sensor behavior
* Sensor importance analysis for identifying critical sensors

**Note:** Dataset is not included due to size.

---

## Architecture Diagram

The system pipeline follows a simple data processing and machine learning workflow:

Data Source → Data Processing → Feature Engineering → Machine Learning → Monitoring Dashboard

Pipeline steps:

1. Raw sensor dataset is loaded and cleaned
2. Feature engineering generates statistical features from sensor data
3. Anomaly detection identifies abnormal sensor behavior
4. Machine learning model predicts potential pump failures
5. Processed data is visualized through an interactive dashboard

---

## Dashboard Screenshot

Example monitoring dashboard:

<img src="images/dashboard.png" width="900">

The dashboard provides:

* Real-time machine health score
* Sensor trend visualization
* Anomaly detection markers
* Failure event visualization
* Important sensor analysis

---

## 🛠️ Tech Stack

Programming Language

* Python

Data Processing

* pandas

Machine Learning

* scikit-learn

Visualization

* matplotlib

Dashboard Framework

* Streamlit

---

## How to Run

### 1. Clone Repository

```
git clone https://github.com/Piest170/pump-monitoring-project.git
cd pump-monitoring-project
```

### 2. Install Dependencies

```
pip install -r requirements.txt
```

### 3. Generate Processed Dataset

Run the preprocessing and machine learning pipeline:

```
python main.py
```

This will generate:

* processed_sensor.csv
* sensor_importance.csv

### 4. Run the Dashboard

```
streamlit run app.py
```

The dashboard will be available at:

```
http://localhost:8501
```

---

## Project Structure

```
pump-monitoring-project

app.py
main.py
reduce_dataset.py
requirements.txt
README.md

sensor_sample.csv
processed_sensor.csv
sensor_importance.csv

images/
  dashboard.png
```

---

## Future Improvements

* Time-series failure prediction
* Advanced anomaly detection models
* Real-time data streaming
* Deployment with cloud infrastructure
