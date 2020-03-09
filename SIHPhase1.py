#!/usr/bin/env python
# coding: utf-8

# In[58]:


import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures


# In[59]:


data1=pd.read_csv('Downloads\MahaData.csv')

data2=data1.copy(deep=True)

data2['Area'].fillna(data2['Area'].median(),inplace=True)
data2['Kg/Hectare'].fillna(data2['Kg/Hectare'].median(),inplace=True)
data2['Rainfall(cm)'].fillna(data2['Rainfall(cm)'].mean(),inplace=True)

#data2['irrigation']=pd.to_numeric(data2['irrigation'],errors='coerce')
#data2['Kg/Hectare']=pd.to_numeric(data2['Kg/Hectare'],errors='coerce')
#data2['Area']=pd.to_numeric(data2['Area'],errors='coerce')

data2.corr(method='pearson')
data2.dropna(axis=1,inplace=True)

columns_list=list(data2.columns)

independentfeatures=sorted(list(set(columns_list)-set(['Kg/Hectare','Production'])))

y=data2['Kg/Hectare'].values
x=data2[independentfeatures].values


# In[27]:


sns.set(style="darkgrid")
sns.regplot(x=data2['Rainfall(cm)'],y=data2['Kg/Hectare'])


# In[28]:


sns.set(style="white")
sns.regplot(x=data2['Area'],y=data2['Kg/Hectare'])


# In[29]:


sns.set(style="ticks")
sns.regplot(x=data2['Year'],y=data2['Kg/Hectare'])3


# In[96]:


train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.2,random_state=0)

poly=PolynomialFeatures(degree = 1)
x_poly=poly.fit_transform(train_x)

poly.fit(x_poly,train_y)
linear2=LinearRegression()
linear2.fit(x_poly,train_y)

prediction_val=linear2.predict(poly.fit_transform(test_x))

print("Predicted values: ")
print(prediction_val)

print(r2_score(prediction_val,test_y))

x2=np.array([4401.625433,83,125,2535,20.59,2019]).reshape(1,6)

prediction_val2=linear2.predict(poly.fit_transform(x2))
print("Predicted value for 2019: ")
print(prediction_val2)


# In[37]:


data2[independentfeatures]


# In[108]:


columns_list=list(data2.columns)

independentfeatures=sorted(list(set(columns_list)-set(['Production'])))

y=data2['Production'].values.reshape(-1,1)
x=data2['Kg/Hectare'].values.reshape(-1,1)


# In[114]:


sns.set(style="ticks")
sns.regplot(x=data2['Kg/Hectare'],y=data2['Production'])


# In[109]:


train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)


# In[113]:


linear=LinearRegression()
linear.fit(train_x,train_y)

prediction_val=linear.predict(test_x)

print("Predicted values: ")
print(prediction_val)

print(r2_score(test_y,prediction_val))

x2=np.array([prediction_val2])
print(x2)

prediction_val21=linear.predict(x2)
print("Predicted value for 2019: ")
print(prediction_val21)


# In[ ]:




