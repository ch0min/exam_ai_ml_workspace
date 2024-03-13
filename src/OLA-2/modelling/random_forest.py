import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

df = pd.read_pickle('../data/cleaned.pkl')


# Splitting the DataFrame
X = df.drop('HeartDisease', axis=1)  # Replace 'HeartDisease' with the actual target column name if different
y = df['HeartDisease']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest model
rf_model = RandomForestClassifier(random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(X_test)
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]  # For ROC-AUC calculation

# Evaluate model - shows accuracy, ROC-AUC score, precision, recall, f1-score
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f'Accuracy: {accuracy}')
print(f'ROC-AUC: {roc_auc}')
print(classification_report(y_test, y_pred))

