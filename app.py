import streamlit as st
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler , OrdinalEncoder , PolynomialFeatures ,RobustScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression , Ridge , Lasso , ElasticNet
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error , r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from category_encoders import BinaryEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVR

st.set_page_config(layout="wide" , page_title="Airbnb APP")

st.title('Airbnb Price Prediction')

column_1 , column_2 , column_3 = st.columns([70,5,70])
with column_1:
    neighbourhood_group=st.selectbox('neighbourhood_group? ',["Manhattan","Brooklyn","Queens","Bronx","Staten Island"])
    room_type=st.selectbox('room_type? ',["Entire home/apt","Private room","Shared room","Hotel room"])
    cancellation_policy=st.selectbox('cancellation_policy? ',["moderate","strict","flexible"])
    Construction_year=st.selectbox('Construction_year? ',["2022","2021","2020","2019","2018","2017","2016","2015",'2014','2013','2012',"2011","2010","2009" ,"2008","2007","2006","2005","2004" ,"2003"])
    host_identity_verified=st.radio('host_identity_verified? ',["unconfirmed","verified"])
    instant_bookable=st.radio("instant_bookable",['False','True'])

with column_3:
    minimum_nights =st.slider('minimum_nights? ',1,30,15)
    number_of_reviews =st.slider('number_of_reviews? ',0,1026,513)
    calculated_host_listings_count =st.slider('calculated_host_listings_count? ',1,332,166)
    availability_365 =st.slider('availability_365? ',0,365,0)
    service_fee =st.slider('service_fee? ',0,240,120)
    review_rate_number =st.slider('review_rate_number? ',1,5,1)

New_Date = pd.DataFrame({'neighbourhood_group':[neighbourhood_group],
                         'room_type':[room_type],
                         'cancellation_policy':[cancellation_policy],
                         'host_identity_verified':[host_identity_verified],
                         'Construction_year':[Construction_year],
                         'instant_bookable':[instant_bookable],
                         'minimum_nights':[minimum_nights],
                         'number_of_reviews':[number_of_reviews],
                         'calculated_host_listings_count':[calculated_host_listings_count],
                         'availability_365':[availability_365],
                         'service_fee':[service_fee],
                         'review_rate_number':[review_rate_number]},index=[0])


transformer=joblib.load('column_Transformer_New.h5')
model=joblib.load('XGBRegressor_new.h5')

Preprocess = transformer.transform(New_Date)
Predict = model.predict(Preprocess)

st.dataframe(New_Date,width=1200,height=10,use_container_width=True)

if st.button('Predict'):
    st.subheader(round(Predict[0],2))
