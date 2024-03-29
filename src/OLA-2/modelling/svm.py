import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)


# Loading train and test files.
df_train = pd.read_pickle("../data/processed/train_data_scaled.pkl")
df_test = pd.read_pickle("../data/processed/test_data_scaled.pkl")

# df_train = pd.read_pickle("../data/processed/train_data_scaled_stratified.pkl")
# df_test = pd.read_pickle("../data/processed/test_data_scaled_stratified.pkl")

# df_train = pd.read_pickle("../data/processed/train_data_scaled_oversampling.pkl")
# df_test = pd.read_pickle("../data/processed/test_data_scaled_oversampling.pkl")

print("Data loaded.")


#############################
# Data Preparation
#############################

# Sampling since SMV takes a long time to evaluate when the dataset is large:
df_train = df_train.sample(frac=0.1, random_state=42)

print("Data sampling complete.")


# Define features and target.
X_train = df_train.drop("HeartDisease", axis=1)
y_train = df_train["HeartDisease"]

X_test = df_test.drop("HeartDisease", axis=1)
y_test = df_test["HeartDisease"]

print("Features and target separated.")


#############################
# Parameter Tuning
#############################




#############################
# Model Training
#############################

print("Training the model...")
svm_model = SVC(random_state=42, class_weight="balanced")
svm_model.fit(X_train, y_train)

print("Training complete.")


#############################
# Evaluation and Cross-Validation
#############################

# 5-fold cross-validation, focusing on F1-score.
# f1_scores = cross_val_score(svm_model, X_train, y_train, cv=3, scoring="f1")

# print("F1 Score for each fold:", f1_scores)

# Calculate the average F1 score:
# average_f1 = f1_scores.mean()
# print("Average F1 score:", average_f1)


#############################
# Model Prediction
#############################

print("Making predictions with the model...")
y_pred = svm_model.predict(X_test)
# y_pred = best_model.predict(X_test)


# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

classification_report_output = classification_report(y_test, y_pred)

print("Prediction complete.")

# Display Results
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("Classification Report:")
print(classification_report_output)
