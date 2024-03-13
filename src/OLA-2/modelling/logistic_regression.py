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

### no scaling or stratified sampling
# df = pd.read_pickle("../data/processed/data_processed.pkl")

# Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(
#     df.drop("HeartDisease", axis=1), df["HeartDisease"], test_size=0.2, random_state=42
# )

### scaled
# df_train = pd.read_pickle("../data/processed/train_data_scaled.pkl")
# df_test = pd.read_pickle("../data/processed/test_data_scaled.pkl")

# X_train, y_train = df_train.drop("HeartDisease", axis=1), df_train["HeartDisease"]
# X_test, y_test = df_test.drop("HeartDisease", axis=1), df_test["HeartDisease"]


### scaled and stratified
df_train = pd.read_pickle("../data/processed/train_data_scaled_stratified.pkl")
df_test = pd.read_pickle("../data/processed/test_data_scaled_stratified.pkl")

X_train, y_train = df_train.drop("HeartDisease", axis=1), df_train["HeartDisease"]
X_test, y_test = df_test.drop("HeartDisease", axis=1), df_test["HeartDisease"]

# Create an instance of Logistic Regression
model = LogisticRegression()

# GridSearch
# param_grid = {"penalty": ["l1", "l2"], "C": [0.001, 0.01, 0.1, 1, 10, 100]}
# grid_search = GridSearchCV(model, param_grid, cv=5)

# # Fit the model to the data
# grid_search.fit(X_train, y_train)

# # Get the best parameters and best score
# best_params = grid_search.best_params_
# best_score = grid_search.best_score_

# print("Best Parameters:", best_params)
# print("Best Score:", best_score)


# Fit the model to the data
# model = grid_search.best_estimator_
model.fit(X_train, y_train)

# Evaluate the model
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
