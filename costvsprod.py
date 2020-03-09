# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 10:45:58 2020

@author: Ashish
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 09:10:06 2020

@author: Ashish
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split 
import numpy as np


data1=pd.read_csv(r'A:\Ashish\TestingProjects\Datasets\Cotton_production_in_india\prodvscp.csv')
data2=data1.copy(deep=True)

data2['Cost Price'].fillna(data2['Cost Price'].median(),inplace=True)

x=data2['Production'].values.reshape(-1,1)
y=data2['Cost Price'].values.reshape(-1,1)

train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,train_size=0.7,random_state=0)


linear=LinearRegression()
linear.fit(train_x,train_y)

prediction_val=linear.predict(test_x)

print("Predicted values: ")
print(prediction_val)

print(r2_score(test_y,prediction_val))



prediction_val2=linear.predict([[5950]])
print("Predicted value for 2019: ")
print(prediction_val2)
