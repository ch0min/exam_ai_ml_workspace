import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Display all columns.
pd.set_option("display.max_columns", 100)

# Loading the dataset.
df = pd.read_csv("../data/raw/heart_2020_cleaned.csv")


###########################
### DATA TRANSFORMATION ###
###########################

### BINARY VARIABLES ###
# Encoding all features which include No/Yes to 0/1.
"""
    HeartDisease, Smoking, AlcoholDrinking, Stroke, DiffWalking,
    Diabetic, PhysicalActivity, Asthma, KidneyDisease, SkinCancer.
    (10)
"""
df = df.replace({"No": 0, "Yes": 1})

# Encoding "Sex" with 0/1.
df["Sex"] = df["Sex"].replace({"Female": 0, "Male": 1})
# df["Sex"].value_counts()

# Encoding "Diabetic" with 0/1, but first we replace everything with No / Yes.
df["Diabetic"] = df["Diabetic"].replace(
    {
        "No, borderline diabetes": "No",
        "Yes (during pregnancy)": "Yes",
    }
)
df["Diabetic"] = df["Diabetic"].replace({"No": 0, "Yes": 1})
# df["Diabetic"].value_counts()


### ORDINAL VARIABLES ###
# Categorizing and Encoding "AgeCategory" into 13 groups.
age_ranges = df["AgeCategory"].unique()
age_codes, _ = pd.factorize(age_ranges, sort=True)
age_range_to_code = dict(zip(age_ranges, age_codes))
df["AgeCategory"] = df["AgeCategory"].replace(age_range_to_code)
# df["AgeCategory"].value_counts().sort_index()


# Categorizing and Encoding "BMI" into 4 different groups.
def categorize_and_encode_bmi(bmi):
    if bmi < 18.5:
        return 0  # Underweight
    elif 18.5 <= bmi < 25.0:
        return 1  # Normal weight
    elif 25.0 <= bmi < 30.0:
        return 2  # Overweight
    else:
        return 3  # Obese


df["BMI"] = df["BMI"].apply(categorize_and_encode_bmi)
# df["BMI"].value_counts()

# Categorizing and Encoding "GenHealth" into 5 different groups.
df["GenHealth"] = df["GenHealth"].replace(
    {"Poor": 0, "Fair": 1, "Good": 2, "Very good": 3, "Excellent": 4}
)
# df["GenHealth"].value_counts()


### NOMINAL VARIABLES ###
# One-hot Encoding "Race" into 6 groups.
"""White, Hispanic, Black, Other, Asian, American Indian/Alskan Native."""
df = pd.get_dummies(df, columns=["Race"])
df.head()
# race_columns = [col for col in df.columns if col.startswith("Race_")]
# race_value_counts = df[race_columns].sum().sort_values(ascending=False)
# race_value_counts


# Data processed to pickle file.
df.to_pickle("../data/processed/data_processed.pkl")
