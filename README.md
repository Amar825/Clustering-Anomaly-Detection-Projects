# Clustering-Anomaly-Detection-Projects
Applying unsupervised clustering and anomaly detection techniques to health-related datasets (heart disease prediction and wearable sensor data).

## Projects

1.  **Heart Disease Prediction (K-Means Clustering)**
    * **Goal:** To group patients into clusters based on medical features (age, sex, cholesterol, max heart rate, etc.) without using pre-defined labels ('heart disease' vs. 'no heart disease'). 
    * **Algorithm:** K-Means Clustering 
    * **Data:** Heart disease dataset (`data/heart-disease.csv`) 
    * **Process:**
        * Load and preprocess data (handle categorical features).
        * Standardize features for equitable distance calculations. 
        * Apply K-Means with `k=2` clusters. 
        * Predict cluster assignments for the data. 
        * (Optional) Evaluate cluster assignments against true labels (for learning purposes). 
    * **File:** `heart_disease_Kmeans.py` 

2.  **Wearable Time-Series Anomaly Detection (KNN)**
    * **Goal:** To detect anomalous (unusually high) overnight resting heart rates (RHR) from smartwatch data, which could indicate early signs of infection or other physiological stress. 
    * **Algorithm:** K-Nearest Neighbors (KNN) for Outlier Detection (via PyOD library) 
    * **Data:** Wearable heart rate data (`data/P100300/Orig_Fitbit_HR.csv`) 
    * **Process:**
        * Load time-series data. 
        * Resample data to get daily median overnight RHR. 
        * Apply KNN outlier detection to identify points significantly distant from their neighbors. 
        * Visualize the time series, highlighting the detected anomalies. [
    * **File:** `wearable_anomaly_knn.py` 

## Requirements

* Python 3.x
* pandas
* scikit-learn
* matplotlib
* pyod (for the wearable data project)
* numpy

Install requirements using pip:
```bash
pip install pandas scikit-learn matplotlib pyod numpy
