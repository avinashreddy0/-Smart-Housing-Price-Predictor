#loading all modules according to requriment

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import accuracy_score,mean_absolute_error,mean_squared_error,r2_score
import streamlit as sts
import joblib


#loading data with pandas 
 
df = pd.read_csv(r"C:\Users\indur\Downloads\house_price_dataset.csv")

#EDA (Exploratory Data Analysis)

print(df.info())
print(df.shape)
print(df.isnull().sum())
print(df.describe())
print(df.head())

# Data Preprocessing

missing_values = df.dropna(subset=['Median_Income','House_Age','Avg_Rooms','Avg_Bedrooms','Population','Avg_Occupancy','Latitude','Longitude','House_Value'])
duplicates = df.drop_duplicates(inplace=True)

#features and target

feature = df[['Median_Income','House_Age','Avg_Rooms','Avg_Bedrooms','Population','Avg_Occupancy','Latitude','Longitude']]

target = df['House_Value']

scale = StandardScaler()
x_scale = scale.fit_transform(feature)

#traing and testing data_set

x_train,x_test,y_train,y_test = train_test_split(x_scale,target,test_size=0.2,random_state=42)

#model 1 :  LINEAR REGRESSION

model = LinearRegression()
model.fit(x_train,y_train)
y_linear_pred = model.predict(x_test)

print('linear_regression:',y_linear_pred)

#saving linear regression ----------------------

joblib.dump(model,'linearregression.pkl')


#model2 = :RANDOMFORESTREGRESSION

model = RandomForestRegressor(n_estimators=10,max_depth=3)
model.fit(x_train,y_train)
y_randam_pred = model.predict(x_test)

print('randomforest:',y_randam_pred)

#saving random forest-----------------

joblib.dump(model,'randomforest.pkl')


#model 3 GRADIENT BOOSTING

model = GradientBoostingRegressor()
model.fit(x_train,y_train)
y_gradient_pred = model.predict(x_test)

print('gradient_prediction:',y_gradient_pred)

#saving gradient boosting 

joblib.dump(model,'gradientboosting.pkl')

#metics :

for y_pred, label in zip(
    [y_linear_pred, y_randam_pred, y_gradient_pred],
    ['LinearRegression', 'RandomForest', 'GradientBoosting']
):
    print(label, y_pred[:5]) 
    print(f"\nModel: {label}")
    print('R2 Score:', r2_score(y_test, y_pred))
    print('MAE:', mean_absolute_error(y_test, y_pred))
    print('MSE:', mean_squared_error(y_test, y_pred))


#saving 

joblib.dump(model,'house_prediction.pkl')



