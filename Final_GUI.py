import streamlit as st
import pandas as pd
import numpy as np
import pickle
import datetime


st.set_page_config(page_title="TrafficPrediction")

hide_streamlit_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Load the pre-trained model
@st.cache_data
def load_model():
    with open('trained_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model


dt_model = load_model()
st.markdown("<p class='header'>Traffic Prediction Model</p>", unsafe_allow_html=True)

# Image for the homepage
st.image("traffic-prediction-using-machine-learning.png", use_column_width=True)


# Function to preprocess input data
def preprocess_input(weekday, weather, time, is_peak_hour, car_count, bike_count, bus_count, truck_count):
    weekday_mapping = {'Monday': 1, 'Tuesday': 5, 'Wednesday': 6, 'Thursday': 4, 'Friday':0, 'Saturday': 3, 'Sunday': 2}
    weather_mapping = {'Sunny': 2, 'Neutral': 1, 'Foggy': 0}
    is_peak_hour_mapping = {False: 0, True: 1}
    
    encoded_weekday = weekday_mapping.get(weekday)
    encoded_weather = weather_mapping.get(weather)
    time_mapping = {
        datetime.time(0, 0): 0, datetime.time(0, 15): 1, datetime.time(0, 30): 2, datetime.time(0, 45): 3,
        datetime.time(1, 0): 4, datetime.time(1, 15): 5, datetime.time(1, 30): 6, datetime.time(1, 45): 7,
        datetime.time(2, 0): 8, datetime.time(2, 15): 9, datetime.time(2, 30): 10, datetime.time(2, 45): 11,
        datetime.time(3, 0): 12, datetime.time(3, 15): 13, datetime.time(3, 30): 14, datetime.time(3, 45): 15,
        datetime.time(4, 0): 16, datetime.time(4, 15): 17, datetime.time(4, 30): 18, datetime.time(4, 45): 19,
        datetime.time(5, 0): 20, datetime.time(5, 15): 21, datetime.time(5, 30): 22, datetime.time(5, 45): 23,
        datetime.time(6, 0): 24, datetime.time(6, 15): 25, datetime.time(6, 30): 26, datetime.time(6, 45): 27,
        datetime.time(7, 0): 28, datetime.time(7, 15): 29, datetime.time(7, 30): 30, datetime.time(7, 45): 31,
        datetime.time(8, 0): 32, datetime.time(8, 15): 33, datetime.time(8, 30): 34, datetime.time(8, 45): 35,
        datetime.time(9, 0): 36, datetime.time(9, 15): 37, datetime.time(9, 30): 38, datetime.time(9, 45): 39,
        datetime.time(10, 0): 40, datetime.time(10, 15): 41, datetime.time(10, 30): 42, datetime.time(10, 45): 43,
        datetime.time(11, 0): 44, datetime.time(11, 15): 45, datetime.time(11, 30): 46, datetime.time(11, 45): 47,
        datetime.time(12, 0): 48, datetime.time(12, 15): 49, datetime.time(12, 30): 50, datetime.time(12, 45): 51,
        datetime.time(13, 0): 52, datetime.time(13, 15): 53, datetime.time(13, 30): 54, datetime.time(13, 45): 55,
        datetime.time(14, 0): 56, datetime.time(14, 15): 57, datetime.time(14, 30): 58, datetime.time(14, 45): 59,
        datetime.time(15, 0): 60, datetime.time(15, 15): 61, datetime.time(15, 30): 62, datetime.time(15, 45): 63,
        datetime.time(16, 0): 64, datetime.time(16, 15): 65, datetime.time(16, 30): 66, datetime.time(16, 45): 67,
        datetime.time(17, 0): 68, datetime.time(17, 15): 69, datetime.time(17, 30): 70, datetime.time(17, 45): 71,
        datetime.time(18, 0): 72, datetime.time(18, 15): 73, datetime.time(18, 30): 74, datetime.time(18, 45): 75,
        datetime.time(19, 0): 76, datetime.time(19, 15): 77, datetime.time(19, 30): 78, datetime.time(19, 45): 79,
        datetime.time(20, 0): 80, datetime.time(20, 15): 81, datetime.time(20, 30): 82, datetime.time(20, 45): 83,
        datetime.time(21, 0): 84, datetime.time(21, 15): 85, datetime.time(21, 30): 86, datetime.time(21, 45): 87,
        datetime.time(22, 0): 88, datetime.time(22, 15): 89, datetime.time(22, 30): 90, datetime.time(22, 45): 91,
        datetime.time(23, 0): 92, datetime.time(23, 15): 93, datetime.time(23, 30): 94, datetime.time(23, 45): 95
    }
    encoded_time = time_mapping.get(time)

    
    encoded_is_peak_hour = is_peak_hour_mapping.get(is_peak_hour)
    
    # Calculate Total
    total = car_count + bike_count + bus_count + truck_count
    
    return [encoded_weekday, encoded_weather,  encoded_time, encoded_is_peak_hour, car_count, bike_count, bus_count, truck_count, total]

# Function to predict traffic situation
def predict_traffic(weekday, weather, time, is_peak_hour, car_count, bike_count, bus_count, truck_count, total):
    # Preprocess input data
    input_data = preprocess_input(weekday, weather, time, is_peak_hour, car_count, bike_count, bus_count, truck_count)
    
    # Make prediction
    prediction = dt_model.predict([input_data])[0]
    
    # Decode the prediction
    traffic_situation_mapping = {2: 'low', 0: 'heavy', 1: 'high', 3: 'normal'}
    traffic_situation = traffic_situation_mapping.get(prediction)
    
    return traffic_situation

# Streamlit UI
st.title('Traffic Prediction')

# User input
weekday = st.selectbox('Select Day of the week', ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
weather = st.selectbox('Select Weather', ['Sunny', 'Neutral', 'Foggy'])
time = st.time_input('Select Time')
is_peak_hour = st.checkbox('Is Peak Hour')
car_count = st.number_input('Enter Car Count', min_value=0)
bike_count = st.number_input('Enter Bike Count', min_value=0)
bus_count = st.number_input('Enter Bus Count', min_value=0)
truck_count = st.number_input('Enter Truck Count', min_value=0)
total = car_count+bus_count+bike_count+truck_count
# Predict button
if st.button('Predict Traffic Situation'):
    traffic_prediction = predict_traffic(weekday, weather, time.strftime('%H:%M:%S'), is_peak_hour, car_count, bike_count, bus_count, truck_count, total)
    st.success(f'Predicted Traffic Situation: {traffic_prediction}')
