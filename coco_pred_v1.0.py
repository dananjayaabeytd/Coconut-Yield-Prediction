# # streamlit_app.py
# import streamlit as st
# import numpy as np
# import pickle

# # Load the model, scaler, and label encoders
# with open('rf_model.pkl', 'rb') as model_file:
#     model = pickle.load(model_file)

# with open('scaler.pkl', 'rb') as scaler_file:
#     scaler = pickle.load(scaler_file)

# with open('label_encoders.pkl', 'rb') as encoders_file:
#     label_encoders = pickle.load(encoders_file)

# # Define a function to predict the yield based on user input
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
    
#     # Make prediction using the loaded model
#     yield_pred = model.predict(inputs_scaled)
#     return yield_pred

# # Streamlit UI
# st.title("Coconut Yield Prediction App")

# # Define Streamlit input widgets for user inputs
# region = st.selectbox("Select Region", label_encoders['Region'].classes_)
# district = st.selectbox("Select District", label_encoders['District'].classes_)
# variety = st.selectbox("Select Variety", label_encoders['Variety'].classes_)
# age = st.slider("Enter Age (Years)", min_value=0, max_value=50, value=10)
# rainfed_irrigated = st.selectbox("Rainfed or Irrigated", label_encoders['Rainfed/Irrigated'].classes_)
# soil_type = st.selectbox("Select Soil Type", label_encoders['Soil Type'].classes_)
# intercropping = st.selectbox("Intercropping", label_encoders['Intercropping'].classes_)
# pest_disease = st.selectbox("Pest/Disease Pressure", label_encoders['Pest/Disease Pressure'].classes_)
# fertilizer = st.slider("Fertilizer Use (kg/tree/year)", min_value=0.0, max_value=10.0, value=1.0)

# # Predict yield based on user inputs
# if st.button("Predict Yield"):
#     prediction = predict_yield(region, district, variety, age, rainfed_irrigated, soil_type, intercropping, pest_disease, fertilizer)
#     st.success(f"Predicted Average Yield (Nuts/Tree/Year): {prediction[0]:.2f}")
