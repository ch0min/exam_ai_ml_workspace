import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


df = pd.read_pickle("../data/processed/data_processed.pkl")


#############################
# Data Preparation
#############################

# Sampling since SMV takes a long time to evaluate when the dataset is large:
# 10# of the data.
df = df.sample(frac=0.2, random_state=42)

# Define features and target.
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

# Splitting the dataset.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardizing the features (data preprocessing/feature scaling).
"""
    Removes Bias: Different features can have different scales 
    (e.g., age might range from 0 to 100, while income might range from thousands to millions). 
    Without standardization, features with larger scales can dominate the model's decision-making process.

    Improves Performance: Many machine learning algorithms, including SVM, 
    perform better when features are on a relatively similar scale.

    Assists in Comparison: Standardization makes the features more comparable and removes the units, 
    so you're not comparing apples and oranges.
"""
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
x_test_scaled = scaler.transform(X_test)


#############################
# Parameter Tuning
#############################



#############################
# Model Training
#############################

svm_model = SVC(random_state=42, class_weight="balanced")
svm_model.fit(X_train, y_train)


#############################
# Evaluation and Cross-Validation
#############################

# 5-fold cross-validation, focusing on F1-score.
f1_scores = cross_val_score(svm_model, X_train_scaled, y_train, cv=3, scoring="f1")

print("F1 Score for each fold:", f1_scores)

# Calculate the average F1 score:
average_f1 = f1_scores.mean()
print("Average F1 score:", average_f1)


#############################
# Model Prediction
#############################

y_pred = svm_model.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

classification_report_output = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("Classification Report:")
print(classification_report_output)


