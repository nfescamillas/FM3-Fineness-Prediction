import streamlit as st
import pandas as pd
import pickle
import xgboost
import numpy as np
import mlflow


st.title('FM3 Fineness Prediction')

main_body_logo ='images/Republic.png'
st.logo(main_body_logo)


with open('model.xgb','rb') as f_in:
    model= pickle.load(f_in)

lst_feedrate= st.number_input('Limestone Feedrate',min_value=0, max_value=10, value= "min")
sep_amps=st.number_input('Air Separator Current',min_value=0, max_value=30, value= "min")
mill_outletp= st.number_input('Mill Outlet Pressure',min_value=-15, max_value=0, value= "min")
mill_outlett=st.number_input('Mill Outlet Temperature',min_value=0, max_value=150, value= "min")
mill_bucket=st.number_input('Mill Bucket Elevator Current',min_value=0, max_value=50, value= "min")
sep_idf=st.number_input('Separator IDF Damper Opening',min_value=0, max_value=100, value= "min")
mill_amps=st.number_input('Mill Current',min_value=200, max_value=300, value= "min")
mill_idf=st.number_input('Mill IDF Damper Opening',min_value=0, max_value=100, value= "min")

features =[{'Limestone Feed Rate':lst_feedrate,
           'Air Separator': sep_amps,
           'Mill Outlet':mill_outletp,
           'Mill Outlet Temperature':mill_outlett,
           'Bucket Elevator':mill_bucket,
           'Separator Inlet Damper':sep_idf,
           'BM3 Main Drive':mill_amps,
           'Mill Dust Fan Damper':mill_idf}]

df= pd.DataFrame(features)
preds = model.predict(df)

st.subheader('FM3 Predicted Fineness 45 um')
st.write(float(preds[0]))