# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 01:48:53 2020

@author: Ashish
"""


import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score 
from sklearn.svm import SVR
def calc5(name,Area):
    data=pd.read_excel('C:\\Users\\aayus\\project\\cotton_prediction\\\COTTON_FULL.xlsx')
    #Area=4000
    data.isnull().sum()
    
    data['State_Name'].unique()
    if name=='tamilnadu':
        tndata=data[data['State_Name']=='Tamil Nadu']
        sns.boxplot(tndata['Area'])
        tndata=tndata[tndata['Area']<=10000]
        sns.boxplot(tndata['Production']>=20000)
        tndata=tndata[tndata['Production']<=20000]
        
        cols=tndata.columns.tolist()
            
        independentfeatures=sorted(list(set(cols)-set(['Production','Cost Price','Crop','District_Name','Selling Price','Season','State_Name'])))
        y=tndata['Production'].values
        x=tndata[independentfeatures].values
            
        x=np.append(x,[[Area,2020]],axis=0)
            
        x_std = preprocessing.scale(x)
            
        userinput=x_std[(x_std.shape[0]-1),:]
        x_std=np.delete(x_std,(x_std.shape[0]-1),axis=0)
            
        train_x,test_x,train_y,test_y=train_test_split(x_std,y,test_size=0.1,random_state=0)
             
        poly=PolynomialFeatures(degree = 2)
        x_poly=poly.fit_transform(train_x)
            
        poly.fit(x_poly,train_y)
        linear2=LinearRegression()
        linear2.fit(x_poly,train_y)
             
        pred=linear2.predict(poly.fit_transform(test_x))            
        print(r2_score(test_y,pred))
        productionpred=linear2.predict(poly.fit_transform(userinput.reshape(1,-1)))
            
        
        x=tndata['Production'].values
        y=tndata['Cost Price'].values
            
        x=np.append(x,productionpred,axis=0)
            
        x_std = preprocessing.scale(x)
            
        userinput=x_std[(x_std.shape[0]-1)]
        x_std=np.delete(x_std,(x_std.shape[0]-1),axis=0)
            
        linear2=LinearRegression()
        linear2.fit(x_std.reshape(-1,1),y.reshape(-1,1))
            
        cppred=linear2.predict(userinput.reshape(1,-1))
            
        data2=pd.read_excel('C:\\Users\\aayus\\project\\cotton_prediction\\costvssp.xlsx')
        
        x=data2[['Year','CPI','Consumption Rate','Cost Price']].values            
        y=data2['SP'].values
            #print(r2_score(test_y,prediction_val)) #0.92
        x=np.append(x,[[2020,7.59,20000,cppred]],axis=0)
            
        x_std = preprocessing.scale(x)
            
        userinput=x_std[(x_std.shape[0]-1),:]
        x_std=np.delete(x_std,(x_std.shape[0]-1),axis=0)
            
        train_x,test_x,train_y,test_y=train_test_split(x_std,y,test_size=0.1,random_state=0)
        
        poly=PolynomialFeatures(degree = 1)
        x_poly=poly.fit_transform(train_x)
            
        poly.fit(x_poly,train_y)
        linear2=LinearRegression()
        linear2.fit(x_poly,train_y)
        
        sppred=linear2.predict(poly.fit_transform(userinput.reshape(1,-1)))
        graphdata=pd.read_excel('C:\\Users\\aayus\\project\\cotton_prediction\\\COTTON_FULL.xlsx',encoding='cp1252')           
        mean2011=graphdata[graphdata['Crop_Year']==2011]
        mean2011=mean2011['Cost Price'].mean()
        
        mean2012=graphdata[graphdata['Crop_Year']==2012 ]
        mean2012=mean2012['Cost Price'].mean()

        
        mean2013=graphdata[graphdata['Crop_Year']==2013 ]
        mean2013=mean2013['Cost Price'].mean()

        mean2010=graphdata[graphdata['Crop_Year']==2010 ]
        mean2010=mean2010['Cost Price'].mean()

        mean2014=graphdata[graphdata['Crop_Year']==2014 ]
        mean2014=mean2014['Cost Price'].mean()
        meanlist=[mean2010,mean2011,mean2012,mean2013,mean2014]
    # production

        mean2011=graphdata[graphdata['Crop_Year']==2011]
        pmean2011=mean2011['Production'].mean()

        mean2012=graphdata[graphdata['Crop_Year']==2012 ]
        pmean2012=mean2012['Production'].mean()

        

        mean2013=graphdata[graphdata['Crop_Year']==2013 ]
        pmean2013=mean2013['Production'].mean()

        mean2010=graphdata[graphdata['Crop_Year']==2010 ]
        pmean2010=mean2010['Production'].mean()

        mean2014=graphdata[graphdata['Crop_Year']==2014 ]
        pmean2014=mean2014['Production'].mean()
        

        pmeanlist=[pmean2010,pmean2011,pmean2012,pmean2013,pmean2014]
        yearlist=graphdata['Crop_Year'].unique()    
        return(cppred,sppred,productionpred,meanlist,yearlist,pmeanlist)
        # return(cppred,sppred)
    elif name=='karnataka':
        kardata=data[data['State_Name']=='Karnataka']
        sns.boxplot(kardata['Area'])
        kardata=kardata[kardata['Area']<60000]
        sns.boxplot(kardata['Production'])
        kardata=kardata[kardata['Production']<=100000]
        
        cols=kardata.columns.tolist()
            
        independentfeatures=sorted(list(set(cols)-set(['Production','Cost Price','Crop','District_Name','Selling Price','Season','State_Name'])))
        y=kardata['Production'].values
        x=kardata[independentfeatures].values
            
        x=np.append(x,[[Area,2020]],axis=0)
            
        x_std = preprocessing.scale(x)
            
        userinput=x_std[(x_std.shape[0]-1),:]
        x_std=np.delete(x_std,(x_std.shape[0]-1),axis=0)
            
        train_x,test_x,train_y,test_y=train_test_split(x_std,y,test_size=0.1,random_state=0)
             
        poly=PolynomialFeatures(degree = 2)
        x_poly=poly.fit_transform(train_x)
            
        poly.fit(x_poly,train_y)
        linear2=LinearRegression()
        linear2.fit(x_poly,train_y)
             
        pred=linear2.predict(poly.fit_transform(test_x))            
        print(r2_score(test_y,pred))
        productionpred=linear2.predict(poly.fit_transform(userinput.reshape(1,-1)))
            
        
        x=kardata['Production'].values
        y=kardata['Cost Price'].values
            
        x=np.append(x,productionpred,axis=0)
            
        x_std = preprocessing.scale(x)
            
        userinput=x_std[(x_std.shape[0]-1)]
        x_std=np.delete(x_std,(x_std.shape[0]-1),axis=0)
            
        linear2=LinearRegression()
        linear2.fit(x_std.reshape(-1,1),y.reshape(-1,1))
            
        cppred=linear2.predict(userinput.reshape(1,-1))
            
        data2=pd.read_excel('C:\\Users\\aayus\\project\\cotton_prediction\\costvssp.xlsx')
        
        x=data2[['Year','CPI','Consumption Rate','Cost Price']].values            
        y=data2['SP'].values
            #print(r2_score(test_y,prediction_val)) #0.92
        x=np.append(x,[[2020,7.59,20000,cppred]],axis=0)
            
        x_std = preprocessing.scale(x)
            
        userinput=x_std[(x_std.shape[0]-1),:]
        x_std=np.delete(x_std,(x_std.shape[0]-1),axis=0)
            
        train_x,test_x,train_y,test_y=train_test_split(x_std,y,test_size=0.1,random_state=0)
        
        poly=PolynomialFeatures(degree = 1)
        x_poly=poly.fit_transform(train_x)
            
        poly.fit(x_poly,train_y)
        linear2=LinearRegression()
        linear2.fit(x_poly,train_y)
        
        sppred=linear2.predict(poly.fit_transform(userinput.reshape(1,-1)))
        graphdata=pd.read_excel('C:\\Users\\aayus\\project\\cotton_prediction\\\COTTON_FULL.xlsx',encoding='cp1252')           
        mean2011=graphdata[graphdata['Crop_Year']==2011]
        mean2011=mean2011['Cost Price'].mean()
        
        mean2012=graphdata[graphdata['Crop_Year']==2012 ]
        mean2012=mean2012['Cost Price'].mean()

        
        mean2013=graphdata[graphdata['Crop_Year']==2013 ]
        mean2013=mean2013['Cost Price'].mean()

        mean2010=graphdata[graphdata['Crop_Year']==2010 ]
        mean2010=mean2010['Cost Price'].mean()

        mean2014=graphdata[graphdata['Crop_Year']==2014 ]
        mean2014=mean2014['Cost Price'].mean()
        meanlist=[mean2010,mean2011,mean2012,mean2013,mean2014]
    # production

        mean2011=graphdata[graphdata['Crop_Year']==2011]
        pmean2011=mean2011['Production'].mean()

        mean2012=graphdata[graphdata['Crop_Year']==2012 ]
        pmean2012=mean2012['Production'].mean()

        

        mean2013=graphdata[graphdata['Crop_Year']==2013 ]
        pmean2013=mean2013['Production'].mean()

        mean2010=graphdata[graphdata['Crop_Year']==2010 ]
        pmean2010=mean2010['Production'].mean()

        mean2014=graphdata[graphdata['Crop_Year']==2014 ]
        pmean2014=mean2014['Production'].mean()
        

        pmeanlist=[pmean2010,pmean2011,pmean2012,pmean2013,pmean2014]
        yearlist=graphdata['Crop_Year'].unique()    
        return(cppred,sppred,productionpred,meanlist,yearlist,pmeanlist)
        
        
