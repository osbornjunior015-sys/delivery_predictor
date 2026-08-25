import streamlit as st
import pickle
import pandas as pd

st.set_page_config(page_title="Delivery Predictor", layout="centered")

@st.cache_resource
def load_model():
    with open('failure_model.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()

st.title("🚚 AI-Powered Failure Prediction System")
st.markdown("Predict if a delivery will fail *before* dispatch.")
st.divider()

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        traffic = st.selectbox("Traffic Level", ['Low', 'Moderate', 'Heavy', 'Gridlocked'])
        weather = st.selectbox("Weather Condition", ['Sunny', 'Rainy', 'Foggy', 'Windy'])
        distance = st.number_input("Distance (km)", min_value=0.5, max_value=100.0, value=10.0)
    with col2:
        experience = st.number_input("Driver Experience (years)", min_value=0, max_value=30, value=5)
        customer_available = st.radio("Is Customer Available?", ('Yes', 'No'))
    
    submitted = st.form_submit_button("Predict Delivery Outcome")

if submitted:
    input_data = pd.DataFrame({
        'traffic_level': [traffic],
        'weather_condition': [weather],
        'distance_km': [distance],
        'driver_experience': [experience],
        'customer_available': [customer_available]
    })
    
    prob = model.predict_proba(input_data)[0][1]
    pred = model.predict(input_data)[0]
    
    st.divider()
    if pred == 1:
        st.error(f"⚠️ HIGH RISK OF FAILURE ({prob*100:.1f}%)")
        st.metric("Recommendation", "Reroute / Call Customer")
    else:
        st.success(f"✅ LOW RISK OF FAILURE ({prob*100:.1f}%)")
        st.metric("Recommendation", "Proceed with Dispatch")
    st.progress(prob, text=f"Risk Level: {prob*100:.0f}%")
