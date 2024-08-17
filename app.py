import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the trained model and preprocessor
with open('diamond_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

# Create the Streamlit app
st.title('Diamond Price Prediction')

# Create input fields for user data
carat = st.number_input('Carat Weight', min_value=0.1, max_value=10.0, value=1.0)
cut = st.selectbox('Cut', ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'])
color = st.selectbox('Color', ['D', 'E', 'F', 'G', 'H', 'I', 'J'])
clarity = st.selectbox('Clarity', ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'])
depth = st.number_input('Depth', min_value=0.1, max_value=100.0, value=60.0)
table = st.number_input('Table', min_value=0.1, max_value=100.0, value=55.0)
x = st.number_input('Length (X)', min_value=0.1, max_value=100.0, value=5.0)
y = st.number_input('Width (Y)', min_value=0.1, max_value=100.0, value=5.0)
z = st.number_input('Depth (Z)', min_value=0.1, max_value=100.0, value=3.0)

# Create a dataframe from user input
input_data = pd.DataFrame({
    'carat': [carat],
    'cut': [cut],
    'color': [color],
    'clarity': [clarity],
    'depth': [depth],
    'table': [table],
    'x': [x],
    'y': [y],
    'z': [z]
})

# Add a prediction button
if st.button('Predict Price'):
    # Preprocess the input data
    processed_data = preprocessor.transform(input_data)
    
    # Make prediction
    prediction = model.predict(processed_data)

    # Display result
    st.subheader('Predicted Diamond Price:')
    st.write(f'${prediction[0]:,.2f}')

# Add some information about the project
st.sidebar.header('About')
st.sidebar.info('This app uses an XGBoost Regressor trained on the Diamond dataset to predict diamond prices based on various features.')