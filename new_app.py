import streamlit as st
import numpy as np
import pickle
from sklearn.preprocessing import PolynomialFeatures

# Load the models, scaler, label encoders, and polynomial features
with open('./models/random_forest_model.pkl', 'rb') as rf_file:
    rf_model = pickle.load(rf_file)

with open('./models/xgboost_model.pkl', 'rb') as xgb_file:
    xgb_model = pickle.load(xgb_file)

with open('./models/gradient_boosting_model.pkl', 'rb') as gbr_file:
    gbr_model = pickle.load(gbr_file)

with open('./helper/scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

with open('./helper/label_encoders.pkl', 'rb') as encoders_file:
    label_encoders = pickle.load(encoders_file)

with open('./helper/polynomial_features.pkl', 'rb') as poly_file:
    poly = pickle.load(poly_file)

# Define a function to predict the yield based on user input
# def predict_yield(region, district, variety, age, rainfed_irrigated, soil_type, intercropping, pest_disease, fertilizer):
#     # Encode inputs using previously saved label encoders
#     inputs = np.array([[region, district, variety, age, rainfed_irrigated, soil_type, intercropping, pest_disease, fertilizer]])
#     inputs[:, 0] = label_encoders['Region'].transform([region])
#     inputs[:, 1] = label_encoders['District'].transform([district])
#     inputs[:, 2] = label_encoders['Variety'].transform([variety])
#     inputs[:, 4] = label_encoders['Rainfed/Irrigated'].transform([rainfed_irrigated])
#     inputs[:, 5] = label_encoders['Soil Type'].transform([soil_type])
#     inputs[:, 6] = label_encoders['Intercropping'].transform([intercropping])
#     inputs[:, 7] = label_encoders['Pest/Disease Pressure'].transform([pest_disease])

#     # Scale the inputs using the saved scaler
#     inputs_scaled = scaler.transform(inputs)
    
#     # Apply polynomial features transformation
#     inputs_poly = poly.transform(inputs_scaled)

#     # Make predictions using the loaded models
#     rf_pred = rf_model.predict(inputs_poly)
#     xgb_pred = xgb_model.predict(inputs_poly)
#     gbr_pred = gbr_model.predict(inputs_poly)

#     # Calculate ensemble prediction
#     ensemble_pred = (rf_pred + xgb_pred + gbr_pred) / 3
#     return ensemble_pred

def predict_yield(region, district, variety, age, rainfed_irrigated, soil_type, intercropping, pest_disease, fertilizer):
    # Encode inputs using previously saved label encoders
    inputs = np.array([[region, district, variety, age, rainfed_irrigated, soil_type, intercropping, pest_disease, fertilizer]])
    inputs[:, 0] = label_encoders['Region'].transform([region])
    inputs[:, 1] = label_encoders['District'].transform([district])
    inputs[:, 2] = label_encoders['Variety'].transform([variety])
    inputs[:, 4] = label_encoders['Rainfed/Irrigated'].transform([rainfed_irrigated])
    inputs[:, 5] = label_encoders['Soil Type'].transform([soil_type])
    inputs[:, 6] = label_encoders['Intercropping'].transform([intercropping])
    inputs[:, 7] = label_encoders['Pest/Disease Pressure'].transform([pest_disease])

    # Apply polynomial features transformation
    inputs_poly = poly.transform(inputs)

    # Scale the inputs using the saved scaler
    inputs_scaled = scaler.transform(inputs_poly)

    # Make predictions using the loaded models
    rf_pred = rf_model.predict(inputs_scaled)
    xgb_pred = xgb_model.predict(inputs_scaled)
    gbr_pred = gbr_model.predict(inputs_scaled)

    # Calculate ensemble prediction
    ensemble_pred = (rf_pred + xgb_pred + gbr_pred) / 3
    return ensemble_pred


# Streamlit UI
st.title("Coconut Yield Prediction App")

# Define Streamlit input widgets for user inputs
region = st.selectbox("Select Region", label_encoders['Region'].classes_)
district = st.selectbox("Select District", label_encoders['District'].classes_)
variety = st.selectbox("Select Variety", label_encoders['Variety'].classes_)
age = st.slider("Enter Age (Years)", min_value=0, max_value=50, value=10)
rainfed_irrigated = st.selectbox("Rainfed or Irrigated", label_encoders['Rainfed/Irrigated'].classes_)
soil_type = st.selectbox("Select Soil Type", label_encoders['Soil Type'].classes_)
intercropping = st.selectbox("Intercropping", label_encoders['Intercropping'].classes_)
pest_disease = st.selectbox("Pest/Disease Pressure", label_encoders['Pest/Disease Pressure'].classes_)
fertilizer = st.slider("Fertilizer Use (kg/tree/year)", min_value=0.0, max_value=10.0, value=1.0)

# Predict yield based on user inputs
if st.button("Predict Yield"):
    prediction = predict_yield(region, district, variety, age, rainfed_irrigated, soil_type, intercropping, pest_disease, fertilizer)
    st.success(f"Predicted Average Yield (Nuts/Tree/Year): {prediction[0]:.2f}")
