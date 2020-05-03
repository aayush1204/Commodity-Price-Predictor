# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 22:45:05 2020

@author: Ashish
"""
import numpy as np
import pandas as pd 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score 
from sklearn import preprocessing

class Maize():
    def calc12(name,Area):
        data1=pd.read_csv('A:\\Ashish\\Cotton-Prediction-master\\Data\\maizedataset.csv',encoding='cp1252')
        
        data1.isnull().sum()
        data1['Selling Price'].fillna(data1['Selling Price'].mean(),inplace=True)
        data1['Production'].fillna(data1['Production'].median(),inplace=True)
        data1['Rainfal (cm)'].fillna(data1['Rainfal (cm)'].median(),inplace=True)
        
        #sns.boxplot(y=data1['Selling Price']) #Box and whiskers plot
        
        data1[data1['Selling Price']>2000].count()
        data1=data1[data1['Selling Price']<2000]
        
        #sns.boxplot(y=data1['Area']) #Box and whiskers plot
        data1[data1['Area']>26500].count()
        data1=data1[data1['Area']<26500]
        
        sns.boxplot(y=data1['Production']) #Box and whiskers plot
        data1[data1['Production']>200000].count()
        data1=data1[data1['Production']<200000]
        
        #Area=5010
            
        data1['State_Name'].unique()
        if name=='Andhra Pradesh':
            apdata=data1[data1['State_Name']=='Andhra Pradesh']
            cols=apdata.columns.tolist()
            independentfeatures=sorted(list(set(cols)-set(['Production','Sunshine','Relative Humidity','Cost Price','Unnamed: 0','Crop','District_Name','Selling Price','Season','State_Name'])))
            y=apdata['Production'].values
            x=apdata[independentfeatures].values
            
            x=np.append(x,[[Area,2020,125,20.59]],axis=0)
            
            x_std = preprocessing.scale(x)
            
            userinput=x_std[140,:]
            x_std=np.delete(x_std,140,axis=0)
            
            train_x,test_x,train_y,test_y=train_test_split(x_std,y,test_size=0.1,random_state=0)
            
            poly=PolynomialFeatures(degree = 1)
            x_poly=poly.fit_transform(train_x)
            
            poly.fit(x_poly,train_y)
            linear2=LinearRegression()
            linear2.fit(x_poly,train_y)
            
            productionpred=linear2.predict(poly.fit_transform(userinput.reshape(1,-1)))
            
            x=apdata['Production'].values
            y=apdata['Cost Price'].values
            
            x=np.append(x,productionpred,axis=0)
            
            x_std = preprocessing.scale(x)
            
            userinput=x_std[140]
            x_std=np.delete(x_std,140,axis=0)
            
            linear2=LinearRegression()
            linear2.fit(x_std.reshape(-1,1),y.reshape(-1,1))
            
            cppred=linear2.predict(userinput.reshape(1,-1))
            
            data2=pd.read_excel('A:\\Ashish\\Cotton-Prediction-master\\Data\\costvssp.xlsx')
        
            x=data2[['Year','CPI','Consumption Rate','Cost Price']].values            
            y=data2['SP'].values
            #print(r2_score(test_y,prediction_val)) #0.92
            x=np.append(x,[[2020,7.59,20000,cppred]],axis=0)
            
            x_std = preprocessing.scale(x)
            
            userinput=x_std[6,:]
            x_std=np.delete(x_std,6,axis=0)
            
            train_x,test_x,train_y,test_y=train_test_split(x_std,y,test_size=0.1,random_state=0)
            
            poly=PolynomialFeatures(degree = 1)
            x_poly=poly.fit_transform(train_x)
            
            poly.fit(x_poly,train_y)
            linear2=LinearRegression()
            linear2.fit(x_poly,train_y)
            
            sppred=linear2.predict(poly.fit_transform(userinput.reshape(1,-1)))
            
            return(cppred,sppred)
            
        elif(name=='Bihar'):
            bihardata=data1[data1['State_Name']=='Bihar']
            cols=bihardata.columns.tolist()
            
            independentfeatures=sorted(list(set(cols)-set(['Crop','District_Name','Selling Price','Cost Price','Season','State_Name'])))
            y=bihardata['Cost Price'].values
            x=bihardata[independentfeatures].values
            train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.2,random_state=0)
            regressor=SVR(kernel='linear')
            regressor.fit(train_x,train_y)
            pred=regressor.predict(test_x)            
            print(r2_score(test_y,pred))   #0.87         
        
        elif (name=='Gujarat'):
            gujdata=data1[data1['State_Name']=='Gujarat']
            cols=gujdata.columns.tolist()
            
            independentfeatures=sorted(list(set(cols)-set(['Crop','District_Name','Production','Season','State_Name'])))
            y=gujdata['Production'].values
            x=gujdata[independentfeatures].values
            train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.2,random_state=0)
            regressor=SVR(kernel='linear')
            regressor.fit(train_x,train_y)
            pred=regressor.predict(test_x)            
            print(r2_score(test_y,pred)) #81.72
        elif name=='Uttar Pradesh':           
            updata=data1[data1['State_Name']=='Uttar Pradesh']
            cols=updata.columns.tolist()
            
            independentfeatures=sorted(list(set(cols)-set(['Crop','District_Name','Production','Season','State_Name'])))
            y=updata['Production'].values
            x=updata[independentfeatures].values
            train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.2,random_state=0)
            regressor=SVR(kernel='linear')
            regressor.fit(train_x,train_y)
            pred=regressor.predict(test_x)            
            print(r2_score(test_y,pred)) #87
        
        elif name=='Tamil Nadu':
            #NO SVR
            tndata=data1[data1['State_Name']=='Tamil Nadu']
            cols=tndata.columns.tolist()
            
            independentfeatures=sorted(list(set(cols)-set(['Crop','District_Name','Production','Season','State_Name'])))
            y=tndata['Production'].values
            x=tndata[independentfeatures].values
            train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.2,random_state=0)
            
            poly=PolynomialFeatures(degree = 2)
            x_poly=poly.fit_transform(train_x)
            
            poly.fit(x_poly,train_y)
            linear2=LinearRegression()
            linear2.fit(x_poly,train_y)
            
            prediction_val=linear2.predict(poly.fit_transform(test_x))
    
            
            print(r2_score(test_y,prediction_val)) #0.92
        elif name=='Karnataka':
            
            kardata=data1[data1['State_Name']=='Karnataka']
            cols=kardata.columns.tolist()
            
            independentfeatures=sorted(list(set(cols)-set(['Crop','District_Name','Production','Season','State_Name'])))
            y=kardata['Production'].values
            x=kardata[independentfeatures].values
            train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.2,random_state=0)
            regressor=SVR(kernel='linear',epsilon=1,degree=3)
            regressor.fit(train_x,train_y)
            pred=regressor.predict(test_x)            
            print(r2_score(test_y,pred)) #0.96 
       
    
    # if request.method=="POST":
    #     forminput=MaizeForm(request.POST)
    #     if forminput.is_valid():
            
    #         Area=forminput.cleaned_data['area']
    #         Region=forminput.cleaned_data['region']
    #         data = Prediction.objects.create(area=Area,region=Region)
    #         data.save()
    #         #!/usr/bin/env python
    #     # coding: utf-8
    #     data1=pd.read_csv(r'C:\Users\aayus\project\cotton_prediction\MahaData.csv')

    #     data2=data1.copy(deep=True)

    #     data2['Area'].fillna(data2['Area'].median(),inplace=True)
    #     data2['Kg/Hectare'].fillna(data2['Kg/Hectare'].median(),inplace=True)
    #     data2['Rainfall(cm)'].fillna(data2['Rainfall(cm)'].mean(),inplace=True)
    #     y=data2['Kg/Hectare']
    #     x1=data2['Area']
    #     #data2['irrigation']=pd.to_numeric(data2['irrigation'],errors='coerce')
    #     #data2['Kg/Hectare']=pd.to_numeric(data2['Kg/Hectare'],errors='coerce')
    #     #data2['Area']=pd.to_numeric(data2['Area'],errors='coerce')

    #     data2.corr(method='pearson')
    #     data2.dropna(axis=1,inplace=True)

    #     columns_list=list(data2.columns)

    #     independentfeatures=sorted(list(set(columns_list)-set(['Kg/Hectare','Production'])))

    #     y=data2['Kg/Hectare'].values
    #     x=data2[independentfeatures].values


    #     sns.set(style="darkgrid")
    #     sns.regplot(x=data2['Rainfall(cm)'],y=data2['Kg/Hectare'])



    #     sns.set(style="white")
    #     sns.regplot(x=data2['Area'],y=data2['Kg/Hectare'])


    #     sns.set(style="ticks")
    #     sns.regplot(x=data2['Year'],y=data2['Kg/Hectare'])


    #     train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.2,random_state=0)

    #     poly=PolynomialFeatures(degree = 1)
    #     x_poly=poly.fit_transform(train_x)

    #     poly.fit(x_poly,train_y)
    #     linear2=LinearRegression()
    #     linear2.fit(x_poly,train_y)

    #     prediction_val=linear2.predict(poly.fit_transform(test_x))

    #     print("Predicted values: ")
    #     print(prediction_val)

    #     print(r2_score(prediction_val,test_y))

    #     x2=np.array([Area,83,125,2535,20.59,2019]).reshape(1,6)

    #     prediction_val2=linear2.predict(poly.fit_transform(x2))
    #     print("Predicted value for 2019: ")
    #     print(prediction_val2) #KG/HECTARE


    #     data2[independentfeatures]


    #     columns_list=list(data2.columns)

    #     independentfeatures=sorted(list(set(columns_list)-set(['Production'])))

    #     y=data2['Production'].values.reshape(-1,1)
    #     x=data2['Kg/Hectare'].values.reshape(-1,1)


    #     sns.set(style="ticks")
    #     sns.regplot(x=data2['Kg/Hectare'],y=data2['Production'])



    #     train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)


    #     linear=LinearRegression()
    #     linear.fit(train_x,train_y)

    #     prediction_val1=linear.predict(test_x)

    #     print("Predicted values: ")
    #     print(prediction_val1)

    #     print(r2_score(test_y,prediction_val1))

    #     x2=np.array([prediction_val2])
    #     print(x2)

    #     prediction_val21=linear.predict(x2)
    #     print("Predicted value for 2019: ")
    #     print( prediction_val21)  #PRODUCTION
                
    #     data1=pd.read_csv(r'C:\Users\aayus\project\cotton_prediction\prodvscp.csv')
    #     data2=data1.copy(deep=True)

    #     data2['Cost Price'].fillna(data2['Cost Price'].median(),inplace=True)
            
    #     x=data2['Production'].values.reshape(-1,1)
    #     y=data2['Cost Price'].values.reshape(-1,1)

    #     train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,train_size=0.7,random_state=0)


    #     linear=LinearRegression()
    #     linear.fit(train_x,train_y)

    #     prediction_val=linear.predict(test_x)

    #     print("Predicted values: ")
    #     print(prediction_val)

    #     print(r2_score(test_y,prediction_val))


    #     prediction_cp=linear.predict([[5950]])
    #     print("Predicted value for 2019: ")
    #     print(prediction_cp)  #COST PRICE

        
    #     data1=pd.read_csv(r'C:\Users\aayus\project\cotton_prediction\Phase2Data.csv')

    #     data2=data1.copy(deep=True)

    #     data2.describe()

    #     data2['CPI'].fillna(data2['CPI'].mean(),inplace=True)
    #     data2['Cost Price'].fillna(data2['Cost Price'].median(),inplace=True)
    #     data2['Selling Price'].fillna(data2['Selling Price'].mean(),inplace=True)

    #     data2.corr(method='pearson')

    #     columns_list=list(data2.columns)

    #     independentfeatures=sorted(list(set(columns_list)-set(['Selling Price'])))

    #     y=data2['Selling Price'].values
    #     x=data2[independentfeatures].values


    #     #sns.set(style="darkgrid")
    #     #0sns.regplot(x=data2['irrigation'],y=data2['Production'])

    #     #sns.set(style="white")
    #     #sns.regplot(x=data2['Area'],y=data2['Production'])

    #     #sns.set(style="ticks")
    #     #sns.regplot(x=data2['Year'],y=data2['Production'])

    #     train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)

    #     poly=PolynomialFeatures(degree = 1)
    #     x_poly=poly.fit_transform(train_x)

    #     poly.fit(x_poly,train_y)
    #     linear2=LinearRegression()
    #     linear2.fit(x_poly,train_y)

    #     prediction_val=linear2.predict(poly.fit_transform(test_x))

    #     print("Predicted values: ")
    #     print(prediction_val)

    #     print(r2_score(prediction_val,test_y))

    #     x2=np.array([15,3000,prediction_cp,4000,2019]).reshape(1,5)

    #     prediction_sp=linear2.predict(poly.fit_transform(x2))
    #     print("Predicted value for 2019: ")
    #     print(prediction_sp)