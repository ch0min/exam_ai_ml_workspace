import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler

df = pd.read_pickle("../data/processed/data_processed.pkl")

# Define features and target.
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

# Splitting the dataset.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardizing the features (data preprocessing/feature scaling).
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Plotting the class distribution BEFORE oversampling:
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
y_train.value_counts().plot(kind="bar", color=["skyblue", "salmon"])
plt.title("Class Distribution Before Oversampling")
plt.xticks(rotation=0)
plt.xlabel("Heart Disease")
plt.ylabel("Frequency")

# Random Oversampling
rd_oversampler = RandomOverSampler(random_state=42)
X_train_oversampled, y_train_oversampled = rd_oversampling = (
    rd_oversampler.fit_resample(X_train_scaled, y_train)
)

# Plotting the class distribution AFTER oversampling:
plt.subplot(1, 2, 2)
pd.Series(y_train_oversampled).value_counts().plot(
    kind="bar", color=["skyblue", "salmon"]
)
plt.title("Class Distribution After Oversampling")
plt.xticks(rotation=0)
plt.xlabel("Heart Disease")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

# Converting the oversampled training data back to a DataFrame:
X_train_oversampled_df = pd.DataFrame(X_train_oversampled, columns=X_train.columns)

# Convert the scaled test features back to a DataFrame
X_test_scaled_df = pd.DataFrame(
    X_test_scaled, index=X_test.index, columns=X_test.columns
)

# Saving the training data with scaled and oversampled features to a pickle file:
train_data_oversampled = pd.concat(
    [
        X_train_oversampled_df,
        pd.DataFrame(y_train_oversampled, columns=["HeartDisease"]),
    ],
    axis=1,
)
train_data_oversampled.to_pickle("../data/processed/train_data_scaled_oversampling.pkl")

# Saving the test data (with scaled features) to a pickle file
test_data_scaled = pd.concat([X_test_scaled_df, y_test], axis=1)
test_data_scaled.to_pickle("../data/processed/test_data_scaled_oversampling.pkl")
