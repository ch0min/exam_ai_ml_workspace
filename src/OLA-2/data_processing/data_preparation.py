import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
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
X_test_scaled = scaler.transform(X_test)


# Convert the scaled training features back to a DataFrame
X_train_scaled_df = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train.columns)

# Convert the scaled test features back to a DataFrame
X_test_scaled_df = pd.DataFrame(X_test_scaled, index=X_test.index, columns=X_test.columns)

# Saving the training data (with scaled features) to a pickle file
train_data_scaled = pd.concat([X_train_scaled_df, y_train], axis=1)
train_data_scaled.to_pickle("../data/processed/train_data_scaled.pkl")

# Saving the test data (with scaled features) to a pickle file
test_data_scaled = pd.concat([X_test_scaled_df, y_test], axis=1)
test_data_scaled.to_pickle("../data/processed/test_data_scaled.pkl")


