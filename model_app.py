import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import PowerTransformer
from sklearn.svm import SVC

#Load the model
load_model = pickle.load(open('wine_svm_final.pkl', 'rb'))

st.title('Simple Red Wine Quality Prediction 🍷')
st.subheader('Please define your parameter below:')
st.write('This simple website application will help to determine your'
         ' red wine quality based on its physicochemical test.'
         'It uses machine learning or AI to predict the quality')

#define the variable by user input
var_1 = st.number_input(label = 'Fixed acidity (g/L)', min_value = 0.00, step = 1.00)
var_2 = st.number_input(label = 'Volatile acidity (g/L)', min_value = 0.00, step = 1.00)
var_3 = st.number_input(label = 'Citric acid (g/L)', min_value = 0.00, step = 1.00)
var_4 = st.number_input(label = 'Residual sugar (g/L)', min_value = 0.00, step = 1.00)
var_5 = st.number_input(label = 'Chlorides (g/L)', min_value = 0.00, step = 1.00)
var_6 = st.number_input(label = 'Free sulfur dioxide (mg/L)', min_value = 0.00, step = 1.00)
var_7 = st.number_input(label = 'Total sulfur dioxide (mg/L)', min_value = 0.00, step = 1.00)
var_8 = st.number_input(label = 'Density (g/cm³)', min_value = 0.000, step = 0.010, format = '%.3f')
var_9 = st.number_input(label = 'pH', min_value = 0.00, step=0.01)
var_10 = st.number_input(label = 'Sulphates (g/L)', min_value = 0.00, step = 1.00)
var_11 = st.number_input(label = 'Alcohol (vol.%)', min_value = 0.00, step = 1.00)
st.write('Total acidity (g/L) = Fixed acidity + volatile acidity')
var_12 = var_1 + var_2
st.info(var_12)
st.write('Bound sulfur dioxide (mg/L)  = Total sulfur dioxide - Free sulfur dioxide')
var_13 = var_7 - var_6
st.info(var_13)

#define the prediction when clicking the button
if st.button('Predict', type='primary') : 
    df = pd.DataFrame({'fixed acidity' : [var_1], 'volatile acidity' : [var_2], 'citric acid' : [var_3], 'residual sugar' : [var_4],                            'chlorides' : [var_5], 'free sulfur dioxide' : [var_6], 'total sulfur dioxide' : [var_7], 
                       'density' : [var_8], 'pH' : [var_9], 'sulphates' : [var_10], 'alcohol' : [var_11], 'total acidity' : [var_12], 
                       'bound sulfur dioxide' : [var_13]})

    new_predict = load_model.predict_proba(df)[:,1] #get the probability
    st.write('User input :', df)
    st.write('Probability prediction :', new_predict)
    if new_predict >= 0.75: #increase the threshold
        st.info('Your wine quality is GOOD')
    else:
        st.error('Your wine quality is BAD')
    
        
    