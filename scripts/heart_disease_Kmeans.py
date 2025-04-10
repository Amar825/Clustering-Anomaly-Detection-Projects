# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt # Although not used for plotting in this specific final script version
import numpy as np

# --- Data Loading and Preprocessing ---

# Load the dataset from the specified path
# Make sure 'heart-disease.csv' is in a 'data' subfolder or adjust the path
df = pd.read_csv("data/heart-disease.csv")

# Transform the categorical 'sex' column to numerical representation
# female: 1, male: 0
df["sex"] = df["sex"].replace({"male": 0, "female": 1})

# --- Feature Selection (Using All Features) ---

# Separate the features (all columns except the last) into X
X = df.iloc[:, :-1].values
# Separate the target variable (the last column) into y
# Although this is unsupervised learning, y (true labels) is kept here
# as per the source document to evaluate the clustering results later.
y = df.iloc[:, -1].values

# --- Data Splitting and Shuffling ---

# Shuffle the data and split it.
# The source document uses train_size=300 and keeps X_train and y_true.
# We use standard Python unpacking to achieve this. random_state ensures reproducibility.
X_train, _, y_true, _ = train_test_split(X, y, random_state=1, train_size=300, test_size=len(df)-300)

# Display the first 5 rows of the training features to verify
print("First 5 rows of training features (X_train):")
print(X_train[:5])

# --- Data Standardization ---

# Initialize the StandardScaler
stdsc = StandardScaler()
# Fit the scaler to the training data and transform X_train
# Standardization ensures features have similar scales for distance-based algorithms like K-Means.
X_train_std = stdsc.fit_transform(X_train)

# --- K-Means Clustering ---

# Initialize the K-Means model
# n_clusters=2: We want to group data into two clusters (e.g., disease vs. no disease)
# init='random': Use random initialization for centroids
# n_init='auto': Automatically determine the number of runs with different centroid seeds
# random_state=0: Ensure reproducibility of the clustering process
km = KMeans(n_clusters=2, init="random", n_init="auto", random_state=0)

# Fit the K-Means model to the standardized training data and predict cluster labels
y_predicted = km.fit_predict(X_train_std)

# --- Evaluation (Comparing with True Labels) ---
# This section compares the unsupervised cluster assignments (y_predicted)
# with the actual labels (y_true) to gauge performance, as done in the source document.

print("\n--- Evaluation ---")
# Display the actual known labels for the first 10 samples in the training set
print("Actual (true) labels for first 10 patients:")
print(y_true[0:10])

# Display the cluster labels assigned by K-Means for the first 10 samples
print("\nPredicted cluster labels for first 10 patients:")
print(y_predicted[0:10])

# --- Basic Comparison Note ---
# Direct comparison assumes cluster 0 aligns with true label 0, and cluster 1 with true label 1.
# K-Means assigns cluster labels (0, 1) arbitrarily; they might be flipped relative to the true labels.
# Proper evaluation often uses metrics like Adjusted Rand Index or involves checking both possible label assignments.
misclassified = np.sum(y_true[0:len(y_predicted)] != y_predicted)
total_predicted = len(y_predicted)
print(f"\nNote: Direct comparison count (assuming cluster 0 == label 0, etc.): {misclassified} out of {total_predicted} potentially misclassified.")
print("Cluster labels (0 or 1) assigned by K-Means might need to be swapped to match true labels (0 or 1) for accurate comparison.")
