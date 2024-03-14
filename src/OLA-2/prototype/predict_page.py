import streamlit as st
import pickle
import pandas as pd
import numpy as np


def load_model():
    with open("../data/processed/heart_disease_predictor.pkl", "rb") as file:
        data = pickle.load(file)
    return data


data = load_model()

regressor_loaded = data["model"]
# binary_variables = data["binary_variables"]
# ordinal_variables = data["ordinal_variables"]
# nominal_variables = data["nominal_variables"]


def binary_variables(df):
    """
    HeartDisease, Smoking, AlcoholDrinking, Stroke, DiffWalking,
    Diabetic, PhysicalActivity, Asthma, KidneyDisease, SkinCancer.
    (10)
    """
    df = df.replace({"No": 0, "Yes": 1})

    # Encoding "Sex" with 0/1.
    df["Sex"] = df["Sex"].replace({"Female": 0, "Male": 1})
    df["Sex"].value_counts()

    # Encoding "Diabetic" with 0/1, but first we replace everything with No / Yes.
    df["Diabetic"] = df["Diabetic"].replace(
        {
            "No, borderline diabetes": "No",
            "Yes (during pregnancy)": "Yes",
        }
    )
    df["Diabetic"] = df["Diabetic"].replace({"No": 0, "Yes": 1})

    return df


def ordinal_variables(df):
    # Categorizing and Encoding "AgeCategory" into 13 groups.
    age_ranges = df["AgeCategory"].unique()
    age_codes, _ = pd.factorize(age_ranges, sort=True)
    age_range_to_code = dict(zip(age_ranges, age_codes))
    df["AgeCategory"] = df["AgeCategory"].replace(age_range_to_code)
    df["AgeCategory"].value_counts().sort_index()

    # Categorizing and Encoding "BMI" into 4 different groups.
    bmi_categories = [
        "Underweight (< 18.5)",
        "Normal weight (18.5 - 25.0)",
        "Overweight (25.0 - 30.0)",
        "Obese (30 <)",
    ]
    bmi_bins = [-np.inf, 18.5, 25.0, 30.0, np.inf]
    df["BMI"] = pd.cut(df["BMI"], bins=bmi_bins, labels=bmi_categories)

    dict_BMI = {category: code for code, category in enumerate(bmi_categories)}
    df["BMI"] = df["BMI"].map(dict_BMI)
    df["BMI"].value_counts()

    # Categorizing and Encoding "GenHealth" into 5 different groups.
    df["GenHealth"] = df["GenHealth"].replace(
        {"Poor": 0, "Fair": 1, "Good": 2, "Very good": 3, "Excellent": 4}
    )

    return df


def nominal_variables(df, training_race_categories):
    if "Race" in df.columns:
        df = pd.get_dummies(df, columns=["Race"])

        # Add missing race categories from the training data
        for category in training_race_categories:
            if f"Race_{category}" not in df.columns:
                df[f"Race_{category}"] = 0
    else:
        st.write("'Race' column does not exist in the DataFrame.")

    return df


def show_predict_page():
    st.title("Heart Disease Prediction")

    data_row = {
        "BMI": 16.6,
        "Smoking": "Yes",
        "AlcoholDrinking": "No",
        "Stroke": "No",
        "PhysicalHealth": 30.0,
        "MentalHealth": 30.0,
        "DiffWalking": "No",
        "Sex": "Female",
        "AgeCategory": "55-59",
        "Race": "White",
        "Diabetic": "Yes",
        "PhysicalActivity": "Yes",
        "GenHealth": "Very good",
        "SleepTime": 5.0,
        "Asthma": "Yes",
        "KidneyDisease": "No",
        "SkinCancer": "Yes",
    }
    df = pd.DataFrame([data_row])

    # Apply the loaded preprocessing functions
    df = binary_variables(df)
    df = ordinal_variables(df)
    training_race_categories = [
        "White",
        "Hispanic",
        "Black",
        "Other",
        "Asian",
        "American Indian/Alaskan Native",
    ]
    df = nominal_variables(df, training_race_categories)

    prediction = regressor_loaded.predict(df)
    st.subheader(f"The predicted outcome is {prediction[0]}")


show_predict_page()
