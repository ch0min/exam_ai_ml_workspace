import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split

df = pd.read_csv('/Users/christoffernielsen/PycharmProjects/exam_ai_ml_workspace/src/OLA-2/data/heart_2020_cleaned.csv')

X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

# Splitting dataset into testing and training

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# df.info()
# df.describe()
# df.head()

y_train_df = y_train.to_frame()

df_training = pd.concat([X_train, y_train_df], axis=1)

df_training.info()

df_training.to_pickle('/Users/christoffernielsen/PycharmProjects/exam_ai_ml_workspace/src/OLA-2/data/training.pkl')