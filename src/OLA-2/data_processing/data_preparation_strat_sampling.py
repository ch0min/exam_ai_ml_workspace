import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler


# Loading processed file.
df = pd.read_pickle("../data/processed/data_processed.pkl")


#############################
# Data Preparation
#############################

# Sampling since SMV takes a long time to evaluate when the dataset is large:
# df = df.sample(frac=0.2, random_state=42)

# Define features and target.
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

# Splitting the dataset.
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# Define StratifiedKFold
stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_index, test_index) in enumerate(stratified_kfold.split(X, y)):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]


# Plotting the class distribution in each fold
fig, axes = plt.subplots(5, 1, figsize=(10, 20))
fig.tight_layout(pad=3.0)

# Looping through each fold
for i, (train_index, test_index) in enumerate(stratified_kfold.split(df, y)):
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Counting the frequency of each class in both train and test sets
    train_counts = y_train.value_counts(normalize=True)
    test_counts = y_test.value_counts(normalize=True)

    # Creating bar plots for each fold
    axes[i].bar(train_counts.index + 0.00, train_counts.values, color='blue', width=0.25, label='Train')
    axes[i].bar(test_counts.index + 0.25, test_counts.values, color='red', width=0.25, label='Test')
    axes[i].set_title(f'Stratified Fold {i+1}')
    axes[i].set_xticks([0, 1])
    axes[i].set_xticklabels(['No Heart Disease', 'Heart Disease'])
    axes[i].legend()

plt.show()


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
X_test_scaled = scaler.transform(X_test)


# Convert the scaled training features back to a DataFrame
X_train_scaled_df = pd.DataFrame(
    X_train_scaled, index=X_train.index, columns=X_train.columns
)

# Convert the scaled test features back to a DataFrame
X_test_scaled_df = pd.DataFrame(
    X_test_scaled, index=X_test.index, columns=X_test.columns
)

# Saving the training data (with scaled features) to a pickle file
train_data_scaled = pd.concat([X_train_scaled_df, y_train], axis=1)
train_data_scaled.to_pickle("../data/processed/train_data_scaled_stratified.pkl")

# Saving the test data (with scaled features) to a pickle file
test_data_scaled = pd.concat([X_test_scaled_df, y_test], axis=1)
test_data_scaled.to_pickle("../data/processed/test_data_scaled_stratified.pkl")
