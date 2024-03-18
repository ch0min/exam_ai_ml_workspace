import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


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


def nominal_variables(df):
    # One-hot Encoding "Race" into 6 groups.
    df = pd.get_dummies(df, columns=["Race"])

    # Add missing columns that the model expects
    expected_columns = [
        "Race_White",
        "Race_Hispanic",
        "Race_Black",
        "Race_Other",
        "Race_Asian",
        "Race_American Indian/Alskan Native",
    ]
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0  # Add missing column with default value 0
    return df


def show_predict_page():
    st.title("Heart Disease Prediction")

    # Dynamic inputs using Streamlit widgets
    bmi = st.slider("BMI", 0.0, 50.0, 25.0)
    smoking = st.radio("Smoking", ["Yes", "No"])
    alcohol_drinking = st.radio("Alcohol Drinking", ["Yes", "No"])
    stroke = st.radio("Stroke", ["Yes", "No"])
    physical_health = st.slider("Physical Health", 0, 30, 0)
    mental_health = st.slider("Mental Health", 0, 30, 0)
    diff_walking = st.radio("Difficulty Walking", ["Yes", "No"])
    sex = st.radio("Sex", ["Female", "Male"])
    age_category = st.selectbox("Age Category", ["18-24", "25-29", "30-34", "35-39", "40-44",
                                                 "45-49", "50-54", "55-59", "60-64", "65-69",
                                                 "70-74", "75-79", "80 or older"])
    race = st.selectbox("Race", ["White", "Hispanic", "Black", "Asian",
                                 "American Indian/Alaskan Native", "Other"])
    diabetic = st.radio("Diabetic", ["Yes", "No", "No, borderline diabetes", "Yes (during pregnancy)"])
    physical_activity = st.radio("Physical Activity", ["Yes", "No"])
    gen_health = st.selectbox("General Health", ["Poor", "Fair", "Good", "Very good", "Excellent"])
    sleep_time = st.slider("Sleep Time", 0, 24, 8)
    asthma = st.radio("Asthma", ["Yes", "No"])
    kidney_disease = st.radio("Kidney Disease", ["Yes", "No"])
    skin_cancer = st.radio("Skin Cancer", ["Yes", "No"])

    data_row = {
        "BMI": bmi,
        "Smoking": smoking,
        "AlcoholDrinking": alcohol_drinking,
        "Stroke": stroke,
        "PhysicalHealth": physical_health,
        "MentalHealth": mental_health,
        "DiffWalking": diff_walking,
        "Sex": sex,
        "AgeCategory": age_category,
        "Race": race,
        "Diabetic": diabetic,
        "PhysicalActivity": physical_activity,
        "GenHealth": gen_health,
        "SleepTime": sleep_time,
        "Asthma": asthma,
        "KidneyDisease": kidney_disease,
        "SkinCancer": skin_cancer,
    }

    df = pd.DataFrame([data_row])

    # Apply the loaded preprocessing functions
    df = binary_variables(df)
    df = ordinal_variables(df)
    df = nominal_variables(df)

    # Predict probabilities
    prediction = regressor_loaded.predict_proba(df)
    percentage = prediction[:, 1] * 100

    if st.button('Predict'):
        st.subheader(f"The predicted outcome is {percentage[0]:.2f}%")


show_predict_page()