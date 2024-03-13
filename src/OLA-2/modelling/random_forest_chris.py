import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Load the processed dataset
df_train = pd.read_pickle("../data/processed/train_data_scaled_stratified.pkl")
df_test = pd.read_pickle("../data/processed/test_data_scaled_stratified.pkl")

# Prepare training and test sets
X_train, y_train = df_train.drop("HeartDisease", axis=1), df_train["HeartDisease"]
X_test, y_test = df_test.drop("HeartDisease", axis=1), df_test["HeartDisease"]

# Initialize and train the random forest classifier with class_weight='balanced'
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Predicting for a new individual
new_instance = {
    'BMI': 2,
    'Smoking': 1,
    'AlcoholDrinking': 1,
    'Stroke': 0,
    'PhysicalHealth': 0,
    'MentalHealth': 0,
    'DiffWalking': 0,
    'Sex': 1,
    'AgeCategory': 4,
    'Diabetic': 1,
    'PhysicalActivity': 1,
    'GenHealth': 3,
    'SleepTime': 8,
    'Asthma': 0,
    'KidneyDisease': 0,
    'SkinCancer': 0,
    'Race': 1
}

# Converting the dictionary to a DataFrame
new_instance_df = pd.DataFrame([new_instance])

# Ensuring that new_instance_df has the same feature order as X_train
new_instance_df = new_instance_df.reindex(columns=X_train.columns, fill_value=0)

# Prediction
prediction_proba = rf_classifier.predict_proba(new_instance_df)

# Extract the probability of having heart disease (class 1)
probability_of_heart_disease = prediction_proba[0][1] * 100
print(f"Based on the inputs, there is a {probability_of_heart_disease:.2f}% chance of having heart disease.")

# Confusion Matrix and Plotting
conf_matrix = confusion_matrix(y_test, y_pred)
conf_matrix_df = pd.DataFrame(conf_matrix, index=['Actual Negative', 'Actual Positive'], columns=['Predicted Negative', 'Predicted Positive'])

plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix_df, annot=True, fmt="d", cmap="Blues", linewidths=.5)
plt.title('Confusion Matrix')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()
