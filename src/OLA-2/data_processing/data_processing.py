import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split

# display all columns
pd.set_option("display.max_columns", 100)

# df = pd.read_csv('/Users/christoffernielsen/PycharmProjects/exam_ai_ml_workspace/src/OLA-2/data/heart_2020_cleaned.csv')
df = pd.read_csv("../data/heart_2020_cleaned.csv")

### Data cleaning
dict_replace = {"No": 0, "Yes": 1}
df = df.replace(dict_replace)

dict_sex = {"Female": 0, "Male": 1}
df = df.replace(dict_sex)

# Diabetic
df = pd.get_dummies(df, prefix="Diabetic", columns=["Diabetic"])


# BMI
def categorize_bmi(bmi):
    if bmi < 18.5:
        return "Underweight (< 18.5)"
    elif 18.5 <= bmi < 25.0:
        return "Normal weight (18.5 - 25.0)"
    elif 25.0 <= bmi < 30.0:
        return "Overweight (25.0 - 30.0)"
    else:
        return "Obese (30 <)"


# Apply the function to the DataFrame
df["BMI"] = df["BMI"].apply(categorize_bmi)

dict_BMI = {
    "Underweight (< 18.5)": 0,
    "Normal weight (18.5 - 25.0)": 1,
    "Overweight (25.0 - 30.0)": 2,
    "Obese (30 <)": 3,
}
df = df.replace(dict_BMI)

# Age Ranges
age_ranges = df["AgeCategory"].unique()

age_codes, _ = pd.factorize(age_ranges, sort=True)

age_range_to_code = dict(zip(age_ranges, age_codes))

df["AgeCategory"] = df["AgeCategory"].map(age_range_to_code)

# General Health

# ordinal encoding:
# Define a mapping from category to numerical value
# genhealth_mapping = {"Poor": 1, "Fair": 2, "Good": 3, "Very good": 4, "Excellent": 5}

# Apply the mapping to the "GenHealth" column
# df["GenHealth"] = df["GenHealth"].map(genhealth_mapping)

# onehot encode
# Create one-hot encoded features for the "GenHealth" column
df = pd.get_dummies(df, prefix="GenHealth", columns=["GenHealth"])

# In the case of the "GenHealth" variable, one-hot encoding may be more appropriate than pd.factorize
# because the categories do not have a clear ordering, and treating them as ordinal variables may not be accurate.
# However, you could experiment with both approaches and see which one works better for your specific use case.

# Race
# Create one-hot encoded features for the "Race" column
df = pd.get_dummies(df, prefix="Race", columns=["Race"])


df.to_pickle("../data/cleaned.pkl")
