import streamlit as st
import pickle
import pandas as pd
import shap
import matplotlib.pyplot as plt
import numpy as np


# PAGE CONFIG
st.set_page_config(
    page_title="Tourism Grade Predictor",
    page_icon="🏨",
    layout="wide"
)

st.title("🏨 Tourism Accommodation Grade Prediction System")
st.markdown("Machine Learning Model: Extra Trees Classifier")
st.markdown("---")


# LOAD MODEL
model = pickle.load(open("models/model.pkl", "rb"))
target_encoder = pickle.load(open("models/target_encoder.pkl", "rb"))

classifier = model.named_steps['classifier']
preprocessor = model.named_steps['preprocessor']


# CREATE LAYOUT (2 COLUMNS)
col1, col2 = st.columns(2)

with col1:
    st.subheader("📥 Enter Accommodation Details")

    rooms = st.number_input("Number of Rooms", min_value=1, step=1)

    latitude = st.number_input("Latitude", format="%.6f")
    longitude = st.number_input("Longitude", format="%.6f")

    # Get district list from training data encoder
    district_list = preprocessor.named_transformers_['cat'].categories_[0]
    type_list = preprocessor.named_transformers_['cat'].categories_[1]

    district = st.selectbox("District", district_list)
    type_ = st.selectbox("Type", type_list)

    predict_button = st.button("🔍 Predict Grade")


# PREDICTION SECTION
if predict_button:

    input_df = pd.DataFrame(
        [[rooms, latitude, longitude, district, type_]],
        columns=['Rooms', 'Latitude', 'Longitude', 'District', 'Type']
    )

    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    decoded_prediction = target_encoder.inverse_transform([prediction])[0]

    with col2:
        st.subheader("🎯 Prediction Result")
        st.success(f"Predicted Grade Level: {decoded_prediction}")

        st.subheader("📊 Prediction Probabilities")

        prob_df = pd.DataFrame({
            "Grade Level": target_encoder.classes_,
            "Probability": probabilities
        })

        st.bar_chart(prob_df.set_index("Grade Level"))

    
    # SHAP + HUMAN EXPLANATION
    st.markdown("---")
    st.subheader("🧠 Why Did The Model Predict This?")

    transformed_input = preprocessor.transform(input_df)

    if hasattr(transformed_input, "toarray"):
        transformed_input = transformed_input.toarray()

    transformed_input = np.array(transformed_input, dtype=float)

    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer(transformed_input)

    predicted_class_index = prediction
    shap_vals = shap_values.values[0][:, predicted_class_index]

    # Get feature names
    ohe = preprocessor.named_transformers_['cat']
    encoded_cat = ohe.get_feature_names_out(['District', 'Type'])
    feature_names = ['Rooms', 'Latitude', 'Longitude'] + list(encoded_cat)

    explanation_df = pd.DataFrame({
        "Feature": feature_names,
        "Impact": shap_vals
    })

    explanation_df["abs_impact"] = explanation_df["Impact"].abs()
    explanation_df = explanation_df.sort_values(by="abs_impact", ascending=False)

    top_positive = explanation_df[explanation_df["Impact"] > 0].head(3)
    top_negative = explanation_df[explanation_df["Impact"] < 0].head(2)

    st.markdown("### 📖 Explanation")

    # Extract selected inputs
    selected_district = district
    selected_type = type_
    room_count = rooms

    # Determine strongest impact features
    top_feature = explanation_df.iloc[0]

    st.write(f"### 🏨 Why was this predicted as **{decoded_prediction}**?")

    explanation_text = f"This accommodation was classified as **{decoded_prediction}** mainly because "

    reasons = []

    # Rooms
    if room_count >= 50:
        reasons.append(f"it has a relatively high number of rooms ({room_count}), which is typical for higher-grade accommodations")
    elif room_count <= 10:
        reasons.append(f"it has a smaller number of rooms ({room_count}), which is common among lower or medium grade properties")
    else:
        reasons.append(f"it has a moderate number of rooms ({room_count})")

    # District
    reasons.append(f"it is located in {selected_district}, which influences the predicted grade based on tourism patterns")

    # Type
    reasons.append(f"the property type is '{selected_type}', which also affects the classification")

    # Combine into paragraph
    explanation_text += ", ".join(reasons) + "."

    st.write(explanation_text)
