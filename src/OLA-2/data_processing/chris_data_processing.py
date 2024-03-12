import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split

df = pd.read_csv(r'C:\Users\chris\PycharmProjects\exam_ai_ml_workspace\src\OLA-2\data\heart_2020_cleaned.csv')

# Replace 'No'/'Yes' with 0/1
dict_replace = {'No': 0, 'Yes': 1}
df = df.replace(dict_replace)

# Replacing outliers in diabetic with 0/1
dict_diabetic = {'No, borderline diabetes': 0, 'Yes (during pregnancy)': 1}
df = df.replace(dict_diabetic)

# Correcting the replacement for sex as it seems it was intended to use dict_sex but mistakenly used dict_replace again
dict_sex = {'Female': 0, 'Male': 1}
df = df.replace(dict_sex)

# Setting the race to 0 if White and 0 if someone else
df['Race'] = df['Race'].apply(lambda x : 0 if x =='White'else 1)
print(df['Race'].value_counts())

# Categorizing the general health
dict_GeneralHealth = {'Excellent': 0, 'Very good' : 1, 'Good' : 2, 'Fair': 3, 'Poor': 4}
df['GenHealth'] = df['GenHealth'].replace(dict_GeneralHealth)

# Categorize and encode BMI within the same column
bmi_categories = ['Underweight (< 18.5)', 'Normal weight (18.5 - 25.0)', 'Overweight (25.0 - 30.0)', 'Obese (30 <)']
bmi_bins = [-np.inf, 18.5, 25.0, 30.0, np.inf]
df['BMI'] = pd.cut(df['BMI'], bins=bmi_bins, labels=bmi_categories)

# Encode BMI categories
dict_BMI = {category: code for code, category in enumerate(bmi_categories)}
df['BMI'] = df['BMI'].map(dict_BMI)

# Handling AgeCategory as in your original script
age_ranges = df["AgeCategory"].unique()
age_codes, _ = pd.factorize(age_ranges, sort=True)
age_range_to_code = dict(zip(age_ranges, age_codes))
df["AgeCategory"] = df["AgeCategory"].map(age_range_to_code)

print(df[["AgeCategory", "BMI"]].head())

df.to_csv("data_processing_tests.csv", index=False)
