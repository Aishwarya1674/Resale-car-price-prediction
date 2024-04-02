
import streamlit as st
import pickle
import numpy as np
import pandas as pd

st.title("Car Price Prediction System")
df = pickle.load(open('df.pkl','rb'))
pipe = pickle.load(open('pipe.pkl','rb'))

#select brand
brand = st.selectbox('Select Brand of your Car',df['brand'].unique(),index=None)

#selecting only cars of selected brand
filt = df['brand'] ==brand
car_filt = df.loc[filt,'full_name'].unique()
car_name = st.selectbox('Select Your Car Name',car_filt,index=None)

#select Regsieterd year
year = st.selectbox('Choose Registered Year',np.sort(df['registered_year'].unique())[::-1],index=None)

#select insurance
insurance_list = list(df['insurance'].unique())
insurance = st.selectbox('Which insurance do you have',insurance_list)

#select transmission of car
filt = df['full_name'] == car_name
transmission_filt = df.loc[filt,'transmission_type'].unique()
transmission = st.selectbox('Automatic vs Manual',transmission_filt)

owner = st.selectbox('Owner Type',df['owner_type'].unique())
fuel = st.selectbox('Fuel Type',df['fuel_type'].unique())

#selecting engines  of selected cars
filt = df['full_name'] == car_name
engine_filt = df.loc[filt,'engine_capacity'].unique()
engine = st.selectbox('Engine Capacity',engine_filt)

kms_driven = st.slider('How Much Car You Drove (Kms Driven)',500,150000)

age = 2023 - year

button_clicked = st.button('predict car price')
if button_clicked:
    result = pipe.predict([[car_name,engine,year,insurance,transmission,kms_driven,owner,fuel,brand,age]])
    result = np.round(np.exp(result))
    st.write('resale price of your car should be :')
    st.write(str(result[0]))



