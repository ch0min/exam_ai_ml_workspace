import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Load the processed dataset
df = pd.read_csv('/Users/christoffernielsen/PycharmProjects/exam_ai_ml_workspace/src/OLA-2/data/data_processing_chris_example.csv')

# Dropping the 'Unnamed: 0' column
if 'Unnamed: 0' in df.columns:
    df.drop('Unnamed: 0', axis=1, inplace=True)

# Select features and target variable
X = df.drop('HeartDisease', axis=1)  # Features: all columns except 'HeartDisease'
y = df['HeartDisease']  # Target variable: 'HeartDisease'

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the random forest classifier with class_weight='balanced'
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# MAKING A NEW INSTANCE WITH CUSTOM INPUTS TO PREDICT PERCENTAGE CHANCE OF GETTING HEART DISEASE

# Predicting for a new individual with the hypothetical encoded values as defined previously
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
    'Race': 1,
     #Race_White
    #'Race_Hispanic': 0,
    #'Race_Black': 0,
    #'Race_Other': 0,
    #'Race_Asian': 0,
    #'Race_American Indian/Alaskan Native': 0
}

# Converting the dictionary to a DataFrame
new_instance_df = pd.DataFrame([new_instance])

# Ensuring that new_instance_df has the same feature order as X_train
new_instance_df = new_instance_df[X_train.columns]

# Prediction
prediction = rf_classifier.predict(new_instance_df)
prediction_proba = rf_classifier.predict_proba(new_instance_df)

# Extract the probability of having heart disease (class 1)
probability_of_heart_disease = prediction_proba[0][1] * 100

print(f"Based on the inputs, there is a {probability_of_heart_disease:.2f}% chance of having heart disease.")

conf_matrix = confusion_matrix(y_test, y_pred)

# Convert confusion matrix to DataFrame for easier plotting
conf_matrix_df = pd.DataFrame(conf_matrix,
                              index=['Actual Negative', 'Actual Positive'],
                              columns=['Predicted Negative', 'Predicted Positive'])

# Plotting using seaborn
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix_df, annot=True, fmt="d", cmap="Blues", linewidths=.5)
plt.title('Confusion Matrix')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()



