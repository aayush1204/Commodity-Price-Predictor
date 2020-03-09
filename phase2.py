# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 06:14:20 2020

@author: Ashish
"""

# -*- coding: utf-8 -*-
"""rre
Spyder Editor

This is a temporary script file.
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 18:10:03 2020

@author: Ashish
"""

import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures

data1=pd.read_csv('A:\Ashish\TestingProjects\Datasets\Cotton_production_in_india\Phase2Data.csv')

data2=data1.copy(deep=True)

data2.describe()

data2['CPI'].fillna(data2['CPI'].mean(),inplace=True)
data2['Cost Price'].fillna(data2['Cost Price'].median(),inplace=True)
data2['Selling Price'].fillna(data2['Selling Price'].mean(),inplace=True)

data2.corr(method='pearson')

columns_list=list(data2.columns)

independentfeatures=sorted(list(set(columns_list)-set(['Selling Price'])))

y=data2['Selling Price'].values
x=data2[independentfeatures].values


#sns.set(style="darkgrid")
#0sns.regplot(x=data2['irrigation'],y=data2['Production'])

#sns.set(style="white")
#sns.regplot(x=data2['Area'],y=data2['Production'])

#sns.set(style="ticks")
#sns.regplot(x=data2['Year'],y=data2['Production'])

train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)

poly=PolynomialFeatures(degree = 1)
x_poly=poly.fit_transform(train_x)

poly.fit(x_poly,train_y)
linear2=LinearRegression()
linear2.fit(x_poly,train_y)

prediction_val=linear2.predict(poly.fit_transform(test_x))

print("Predicted values: ")
print(prediction_val)

print(r2_score(prediction_val,test_y))

x2=np.array([15,3000,100,4000,2019]).reshape(1,5)

prediction_val2=linear2.predict(poly.fit_transform(x2))
print("Predicted value for 2019: ")
print(prediction_val2)
