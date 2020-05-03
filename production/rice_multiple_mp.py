#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
sns.set()


def calc3(Area):

    data = pd.read_csv('C:\\Users\\aayus\\project\\cotton_prediction\\Rice_MP.csv')

    data = data[data['Production'] < 90000]
    data.describe()

    # f, (ax1, ax2, ax3) = plt.subplots(1,3, sharey= True, figsize=(15,3))
    # ax1.scatter(data['Area'], data['Temp'])
    # ax1.set_title('Area and Temp')
    # ax2.scatter(data['Area'], data['Rainfall(cm)'])
    # ax2.set_title('Area and Rainfall')
    # ax3.scatter(data['Temp'], data['Rainfall(cm)'])
    # ax3.set_title('Temp and Rainfall')
    # plt.show()


    # from statsmodels.stats.outliers_influence import variance_inflation_factor

    # vars = data [['Area', 'Temp', 'Rainfall(cm)']]
    # vif = pd.DataFrame()
    # vif['VIF'] = [variance_inflation_factor(vars.values, i) for i in range (vars.shape[1])]
    # vif["features"] = vars.columns


    # from numpy import cov
    # covariance = cov(data['Temp'], data['Rainfall(cm)'])
    # # print(covariance)



    y = data['Production']

    x = data [['Area', 'Crop_Year', 'Rainfall(cm)', 'Temp']]
    x=np.append(x,[[Area, 2020, 0.22, 23]], axis=0)
    #x.reshape(-1,1)
    x.shape

    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()


    scaler.fit(x)

    x_scaled = scaler.transform(x)
    userinput = x_scaled[110,:]
    x_scaled = np.delete(x_scaled, 110, axis=0)

    x_scaled.shape

    from sklearn.model_selection import train_test_split

    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=420)

    reg = LinearRegression()
    reg.fit(x_train, y_train)

    r2 = reg.score(x_train, y_train)

    x_train.shape
    n = x.shape[0]
    p = x.shape[1]
    adj_r2 = 1 - (1-r2)*(n-1)/ (n-p-1)

    # adj_r2
    # r2
    y_train.shape

    from sklearn.metrics import r2_score
    productionpred = reg.predict(userinput.reshape(1,-1))

    data = pd.read_csv('C:\\Users\\aayus\\project\\cotton_prediction\\Rice_MP.csv')
    x=data['Production'].values
    y=data['Cost Price'].values
    
    x=np.append(x,productionpred,axis=0)
    
    x_std = preprocessing.scale(x)
    
    userinput=x_std[95]
    x_std=np.delete(x_std,95,axis=0)
    
    linear2=LinearRegression()
    linear2.fit(x_std.reshape(-1,1),y.reshape(-1,1))
    
    cppred=linear2.predict(userinput.reshape(1,-1))
    
    data2=pd.read_csv('C:\\Users\\aayus\\project\\cotton_prediction\\Rice_MP.csv')

    # x=data2[['Year','CPI','Consumption Rate','Cost Price']].values            
    # y=data2['SP'].values
    x=data2[['Crop_Year','CPI','Consumption','Cost Price']].values            
    y=data2['SELLING P'].values
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
    
        
    graphdata=pd.read_csv('C:\\Users\\aayus\\project\\cotton_prediction\\Rice_MP.csv',encoding='cp1252')
    mean2011=graphdata[graphdata['Crop_Year']==2011]
    mean2011=mean2011['Cost Price'].mean()
    
    mean2012=graphdata[graphdata['Crop_Year']==2012 ]
    mean2012=mean2012['Cost Price'].mean()

    
    mean2013=graphdata[graphdata['Crop_Year']==2013 ]
    mean2013=mean2013['Cost Price'].mean()

    mean2010=graphdata[graphdata['Crop_Year']==2010 ]
    mean2010=mean2010['Cost Price'].mean()

    mean2009=graphdata[graphdata['Crop_Year']==2009 ]
    mean2009=mean2009['Cost Price'].mean()
    meanlist=[mean2009,mean2010,mean2011,mean2012,mean2013,]
# production

    mean2011=graphdata[graphdata['Crop_Year']==2011]
    pmean2011=mean2011['Production'].mean()

    mean2012=graphdata[graphdata['Crop_Year']==2012 ]
    pmean2012=mean2012['Production'].mean()

    

    mean2013=graphdata[graphdata['Crop_Year']==2013 ]
    pmean2013=mean2013['Production'].mean()

    mean2010=graphdata[graphdata['Crop_Year']==2010 ]
    pmean2010=mean2010['Production'].mean()

    mean2009=graphdata[graphdata['Crop_Year']==2009 ]
    pmean2009=mean2009['Production'].mean()
    

    pmeanlist=[pmean2009,pmean2010,pmean2011,pmean2012,pmean2013,]
    yearlist=graphdata['Crop_Year'].unique()    
    return(cppred,sppred,productionpred,meanlist,yearlist,pmeanlist)
    