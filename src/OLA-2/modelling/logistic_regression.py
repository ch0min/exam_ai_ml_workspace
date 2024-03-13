import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score, recall_score

pd.set_option("display.max_columns", 100)
# df = pd.read_pickle("../data/cleaned.pkl")

# Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(
#     df.drop("HeartDisease", axis=1), df["HeartDisease"], test_size=0.2, random_state=42
# )
df_train = pd.read_pickle("../data/processed/train_data_scaled.pkl")
df_test = pd.read_pickle("../data/processed/test_data_scaled.pkl")

X_train, y_train = df_train.drop("HeartDisease", axis=1), df_train["HeartDisease"]
X_test, y_test = df_test.drop("HeartDisease", axis=2), df_test["HeartDisease"]


# X, y = df.drop("HeartDisease", axis=1), df["HeartDisease"]

# Create an instance of Logistic Regression
model = LogisticRegression()

# Fit the model to the data
model.fit(X_train, y_train)


# Evaluate the model
accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)
# 0.9137572507387545

# Generate predictions
y_pred = model.predict(X_test)
precision = precision_score(y_test, y_pred)
# 0.5363984674329502
recall = recall_score(y_test, y_pred)
# 0.10014306151645208


print("Precision:", precision)
print("Recall:", recall)

# confusion matrix
confusion_matrix(y_test, y_pred)
# [[57883,   484],
# [ 5032,   560]]

# calculate f1 score
f1 = f1_score(y_test, y_pred)
print("F1 score:", f1)
# 0.16877637130801687


# Test data
test_data = pd.DataFrame(
    {
        "BMI": [3],
        "Smoking": [1],
        "AlcoholDrinking": [0],
        "Stroke": [0],
        "PhysicalHealth": [30.0],
        "MentalHealth": [30.0],
        "DiffWalking": [0],
        "Sex": [1],
        "AgeCategory": [11],
        "PhysicalActivity": [1],
        "SleepTime": [9],
        "Asthma": [1],
        "KidneyDisease": [1],
        "SkinCancer": [0],
        "Diabetic_0": [0],
        "Diabetic_1": [0],
        "Diabetic_No, borderline diabetes": [0],
        "Diabetic_Yes (during pregnancy)": [1],
        "GenHealth": [0],
        "Race_American Indian/Alaskan Native": [0],
        "Race_Asian": [0],
        "Race_Black": [0],
        "Race_Hispanic": [0],
        "Race_Other": [0],
        "Race_White": [1],
    }
)

# Generate predictions
predictions = model.predict(test_data)
print(predictions)
