# ==============================================================================
# PROJECT: Foundational AI & Data Analysis for Computational Oncology
# DESCRIPTION: Exploratory Data Analysis (EDA) and Machine Learning Classification
#              focused on identifying patterns in gene expression / cancer data.
# AUTHOR: [Azam Jannesari]
# DATE: June 2026
# ==============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ------------------------------------------------------------------------------
# STEP 1: Data Simulation / Data Loading
# Mocking a high-throughput gene expression dataset (e.g., Cancer vs. Normal cells)
# ------------------------------------------------------------------------------
print("--> Step 1: Loading and initializing the dataset...")
np.random.seed(42)
num_samples = 200
num_features = 10  # Representing 10 specific genetic markers

# Generating synthetic expression profiles for cancer (1) and healthy (0) samples
X_data = np.random.randn(num_samples, num_features)
# Introducing a synthetic biological correlation for the target label
y_data = np.where(X_data[:, 0] + X_data[:, 1] > 0.5, 1, 0)

# Structuring the data into a clean Pandas DataFrame
feature_names = [f"Gene_Marker_{i+1}" for i in range(num_features)]
df = pd.DataFrame(X_data, columns=feature_names)
df['Tumor_Class'] = y_data

print(f"Dataset successfully initialized with shape: {df.shape}\n")

# ------------------------------------------------------------------------------
# STEP 2: Exploratory Data Analysis (EDA) & Quality Control
# ------------------------------------------------------------------------------
print("--> Step 2: Performing Exploratory Data Analysis (EDA)...")
# Checking for missing values (Essential step for biological data quality control)
missing_values = df.isnull().sum().sum()
print(f"Total missing values in dataset: {missing_values}")

# Displaying basic statistical properties of the expression matrix
print("\nStatistical Summary of Genetic Features:")
print(df.describe().round(3))

# ------------------------------------------------------------------------------
# STEP 3: Data Preprocessing & Train-Test Split
# ------------------------------------------------------------------------------
print("\n--> Step 3: Preprocessing data and partitioning into train/test sets...")
X = df.drop('Tumor_Class', axis=1)
y = df['Tumor_Class']

# Standardizing features (Z-score normalization) for optimal algorithmic performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 80-20 stratified split to preserve class proportions
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.20, stratify=y, random_state=42
)
print(f"Training set shape: {X_train.shape} | Testing set shape: {X_test.shape}\n")

# ------------------------------------------------------------------------------
# STEP 4: Machine Learning Model Training (Random Forest)
# ------------------------------------------------------------------------------
print("--> Step 4: Training the Random Forest Classifier...")
# Utilizing Random Forest due to its robust handling of high-dimensional bio-data
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
print("Model training completed successfully.\n")

# ------------------------------------------------------------------------------
# STEP 5: Model Evaluation & Performance Metrics
# ------------------------------------------------------------------------------
print("--> Step 5: Evaluating model performance on unseen test data...")
y_pred = rf_model.predict(X_test)

# Calculating comprehensive classification metrics
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Overall Model Accuracy: {accuracy * 100:.2f}%")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nDetailed Classification Report:")
print(class_report)

# ------------------------------------------------------------------------------
# STEP 6: Feature Importance Analysis (Biological Insights)
# ------------------------------------------------------------------------------
print("--> Step 6: Extracting feature importances for genetic markers...")
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

print("Top 5 Most Discriminating Gene Markers:")
for f in range(5):
    print(f"{f+1}. {feature_names[indices[f]]} (Importance: {importances[indices[f]]:.4f})")

print("\n==============================================================================")
print("Execution Completed. Pipeline ran successfully without any exceptions.")
print("==============================================================================")
