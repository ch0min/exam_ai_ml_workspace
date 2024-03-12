import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the processed dataset
df = pd.read_csv(r"C:\Users\chris\PycharmProjects\exam_ai_ml_workspace\src\OLA-2\data_processing\data_processed_fdsas.csv")

# Drop the 'Unnamed: 0' column if it exists to ensure it does not interfere with model training and prediction
if 'Unnamed: 0' in df.columns:
    df.drop('Unnamed: 0', axis=1, inplace=True)

# Select features and target variable
X = df.drop('HeartDisease', axis=1)  # Features: all columns except 'HeartDisease'
y = df['HeartDisease']  # Target variable: 'HeartDisease'

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the random forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Example: Predicting for a new individual with the hypothetical encoded values as defined previously
new_instance = {
    'BMI': 3,  # Assume 2: Overweight
    'Smoking': 1,
    'AlcoholDrinking': 1,
    'Stroke': 1,
    'PhysicalHealth': 3,
    'MentalHealth': 2,
    'DiffWalking': 1,
    'Sex': 1,  # Assume 1: Male
    'AgeCategory': 6,  # Assuming AgeCategory code 6 corresponds to the desired age range
    'Diabetic': 1,
    'PhysicalActivity': 1,
    'GenHealth': 3,  # Assume 3: Very good
    'SleepTime': 7,
    'Asthma': 0,
    'KidneyDisease': 0,
    'SkinCancer': 0,
    'Race_White': 1,
    'Race_Hispanic': 0,
    'Race_Black': 0,
    'Race_Other': 0,
    'Race_Asian': 0,
    'Race_American Indian/Alaskan Native': 0
}

# Convert the dictionary to a DataFrame
new_instance_df = pd.DataFrame([new_instance])

# Ensure that new_instance_df has the same feature order as X_train
new_instance_df = new_instance_df[X_train.columns]

# Prediction
prediction = rf_classifier.predict(new_instance_df)
prediction_proba = rf_classifier.predict_proba(new_instance_df)

# Extract the probability of having heart disease (class 1)
probability_of_heart_disease = prediction_proba[0][1] * 100  # Convert to percentage

# Print the probability as a percentage
print(f"Based on the inputs, there is a {probability_of_heart_disease:.2f}% chance of having heart disease.")

