import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    classification_report,
    precision_score,
    precision_recall_curve,
    roc_curve,
)
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", 100)

# no scaling or stratified sampling
# df = pd.read_pickle("../data/processed/data_processed.pkl")

# Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(
#     df.drop("HeartDisease", axis=1), df["HeartDisease"], test_size=0.2, random_state=42
# )

# scaled
# df_train = pd.read_pickle("../data/processed/train_data_scaled.pkl")
# df_test = pd.read_pickle("../data/processed/test_data_scaled.pkl")

# X_train, y_train = df_train.drop("HeartDisease", axis=1), df_train["HeartDisease"]
# X_test, y_test = df_test.drop("HeartDisease", axis=1), df_test["HeartDisease"]


# scaled
df_train = pd.read_pickle("../data/processed/train_data_scaled_stratified.pkl")
df_test = pd.read_pickle("../data/processed/test_data_scaled_stratified.pkl")

X_train, y_train = df_train.drop("HeartDisease", axis=1), df_train["HeartDisease"]
X_test, y_test = df_test.drop("HeartDisease", axis=1), df_test["HeartDisease"]


# Create an instance of Logistic Regression
model = LogisticRegression(class_weight="balanced")

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


# Create an instance of Logistic Regression
# Define the parameter grid
param_grid = {"penalty": ["l1", "l2"], "C": [0.001, 0.01, 0.1, 1, 10, 100]}

# Create an instance of Logistic Regression
model = LogisticRegression()

# Create GridSearchCV object
# grid_search = GridSearchCV(model, param_grid, cv=5)

# # Fit the model to the data
# grid_search.fit(X_train, y_train)

# # Get the best parameters and best score
# best_params = grid_search.best_params_
# best_score = grid_search.best_score_

# print("Best Parameters:", best_params)
# print("Best Score:", best_score)

# Fit the model to the data
model.fit(X_train, y_train)
# model = grid_search.best_estimator_


# Get the predicted probabilities
# y_scores = model.predict_proba(X_train)[:, 1]

# # Get precision and recall values for different thresholds
# precisions, recalls, thresholds = precision_recall_curve(y_train, y_scores)

# # Find the threshold that gives you the best recall
# # This will depend on how much you want to prioritize recall over precision
# # For example, you might choose the threshold that gives you a recall > 0.8
# idx = next(i for i, recall in enumerate(recalls) if recall < 0.8) - 1
# best_threshold = thresholds[idx]

# Use this threshold to convert the probabilities into class predictions
# y_pred = (y_scores > best_threshold).astype(int)
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))

# roc curve
# Calculate the predicted probabilities
y_scores = model.predict_proba(X_test)[:, 1]

# Calculate the false positive rate, true positive rate, and thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_scores)

# Plot the ROC curve
plt.plot(fpr, tpr)
plt.plot(
    [0, 1], [0, 1], linestyle="--", color="r"
)  # Add the dotted line for random classifier
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.show()
