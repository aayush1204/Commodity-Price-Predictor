from django.shortcuts import render
import requests
from .forms import  CommodityForm, RiceForm, MaizeForm, CottonForm
import pandas as pd
from django.http import HttpResponse
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from .models import Prediction
from django.views.generic import TemplateView
from datetime import datetime

from production.cotton_katn import calc5
from production.rice_multiple_mp import calc3
from production.rice_multiple_odwb import calc4
from sklearn.svm import SVR
from .models import Prediction

from sklearn import preprocessing
# from production import maizereg
# Create your views here.
def calc12(name,Area,sb):
        if sb == 1:
            data1=pd.read_csv('C:\\Users\\aayus\\project\\cotton_prediction\\maizedataset.csv',encoding='cp1252')
            
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
            if name=='andhrapradesh':
                apdata=data1[data1['State_Name']=='Andhra Pradesh']
                cols=apdata.columns.tolist()
                independentfeatures=sorted(list(set(cols)-set(['Production','Sunshine','Relative Humidity','Cost Price','Unnamed: 0','Crop','District_Name','Selling Price','Season','State_Name'])))
                y=apdata['Production'].values
                x=apdata[independentfeatures].values
                
                x=np.append(x,[[Area,2020,125,20.59]],axis=0)
                
                x_std = preprocessing.scale(x)
                
                userinput=x_std[95,:]
                x_std=np.delete(x_std,95,axis=0)
                
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
                
                userinput=x_std[95]
                x_std=np.delete(x_std,95,axis=0)
                
                linear2=LinearRegression()
                linear2.fit(x_std.reshape(-1,1),y.reshape(-1,1))
                
                cppred=linear2.predict(userinput.reshape(1,-1))
                
                data2=pd.read_excel('C:\\Users\\aayus\\project\\cotton_prediction\\costvssp.xlsx')
            
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
                
                if sb == 1:    
                    graphdata=pd.read_csv('C:\\Users\\aayus\\project\\cotton_prediction\\maizedataset.csv',encoding='cp1252')
                mean2011=graphdata[(graphdata['Crop_Year']==2011) & (graphdata['State_Name']=='Andhra Pradesh')]
                mean2011=mean2011['Cost Price'].mean()

                mean2012=graphdata[(graphdata['Crop_Year']==2012) & (graphdata['State_Name']=='Andhra Pradesh')]
                mean2012=mean2012['Cost Price'].mean()

                mean2014=graphdata[(graphdata['Crop_Year']==2014) & (graphdata['State_Name']=='Andhra Pradesh')]
                mean2014=mean2014['Cost Price'].mean()

                mean2013=graphdata[(graphdata['Crop_Year']==2013) & (graphdata['State_Name']=='Andhra Pradesh')]
                mean2013=mean2013['Cost Price'].mean()

                mean2010=graphdata[(graphdata['Crop_Year']==2010) & (graphdata['State_Name']=='Andhra Pradesh')]
                mean2010=mean2010['Cost Price'].mean()

                mean2009=graphdata[(graphdata['Crop_Year']==2009) & (graphdata['State_Name']=='Andhra Pradesh')]
                mean2009=mean2009['Cost Price'].mean()
                meanlist=[mean2009,mean2010,mean2011,mean2012,mean2013,mean2014]
            # production

                mean2011=graphdata[(graphdata['Crop_Year']==2011) & (graphdata['State_Name']=='Andhra Pradesh')]
                pmean2011=mean2011['Production'].mean()

                mean2012=graphdata[(graphdata['Crop_Year']==2012) & (graphdata['State_Name']=='Andhra Pradesh')]
                pmean2012=mean2012['Production'].mean()

                mean2014=graphdata[(graphdata['Crop_Year']==2014) & (graphdata['State_Name']=='Andhra Pradesh')]
                pmean2014=mean2014['Production'].mean()

                mean2013=graphdata[(graphdata['Crop_Year']==2013) & (graphdata['State_Name']=='Andhra Pradesh')]
                pmean2013=mean2013['Production'].mean()

                mean2010=graphdata[(graphdata['Crop_Year']==2010) & (graphdata['State_Name']=='Andhra Pradesh')]
                pmean2010=mean2010['Production'].mean()

                mean2009=graphdata[(graphdata['Crop_Year']==2009) & (graphdata['State_Name']=='Andhra Pradesh') ]
                pmean2009=mean2009['Production'].mean()
                
                pmeanlist=[pmean2009,pmean2010,pmean2011,pmean2012,pmean2013,pmean2014]
                yearlist=graphdata['Crop_Year'].unique()    
                return(cppred,sppred,productionpred,meanlist,yearlist,pmeanlist)
                
            elif(name=='bihar'):
                bihardata=data1[data1['State_Name']=='Bihar']
                cols=bihardata.columns.tolist()
                
                independentfeatures=sorted(list(set(cols)-set(['Production','Sunshine','Relative Humidity','Cost Price','Unnamed: 0','Crop','District_Name','Selling Price','Season','State_Name'])))
                y=bihardata['Production'].values
                x=bihardata[independentfeatures].values
                
                x=np.append(x,[[Area,2020,125,20.59]],axis=0)
                
                x_std = preprocessing.scale(x)
                
                userinput=x_std[195,:]
                x_std=np.delete(x_std,195,axis=0)
                
                train_x,test_x,train_y,test_y=train_test_split(x_std,y,test_size=0.1,random_state=0)
                
                poly=PolynomialFeatures(degree = 1)
                x_poly=poly.fit_transform(train_x)
                
                poly.fit(x_poly,train_y)
                linear2=LinearRegression()
                linear2.fit(x_poly,train_y)
                
                pred=linear2.predict(poly.fit_transform(test_x))            
                print(r2_score(test_y,pred))
                productionpred=linear2.predict(poly.fit_transform(userinput.reshape(1,-1)))
                
                x=bihardata['Production'].values
                y=bihardata['Cost Price'].values
                
                x=np.append(x,productionpred,axis=0)
                
                x_std = preprocessing.scale(x)
                
                userinput=x_std[195]
                x_std=np.delete(x_std,195,axis=0)
                
                linear2=LinearRegression()
                linear2.fit(x_std.reshape(-1,1),y.reshape(-1,1))
                
                cppred=linear2.predict(userinput.reshape(1,-1))
                
                data2=pd.read_excel('C:\\Users\\aayus\\project\\cotton_prediction\\costvssp.xlsx')
            
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
                
                if sb == 1:    
                    graphdata=pd.read_csv('C:\\Users\\aayus\\project\\cotton_prediction\\maizedataset.csv',encoding='cp1252')
                mean2011=graphdata[(graphdata['Crop_Year']==2011) & (graphdata['State_Name']=='Bihar')]
                mean2011=mean2011['Cost Price'].mean()

                mean2012=graphdata[(graphdata['Crop_Year']==2012) & (graphdata['State_Name']=='Bihar') ]
                mean2012=mean2012['Cost Price'].mean()

                mean2014=graphdata[(graphdata['Crop_Year']==2014) & (graphdata['State_Name']=='Bihar') ]
                mean2014=mean2014['Cost Price'].mean()

                mean2013=graphdata[(graphdata['Crop_Year']==2013) & (graphdata['State_Name']=='Bihar')]
                mean2013=mean2013['Cost Price'].mean()

                mean2010=graphdata[(graphdata['Crop_Year']==2010) & (graphdata['State_Name']=='Bihar')]
                mean2010=mean2010['Cost Price'].mean()

                mean2009=graphdata[(graphdata['Crop_Year']==2009) & (graphdata['State_Name']=='Bihar')]
                mean2009=mean2009['Cost Price'].mean()
                meanlist=[mean2009,mean2010,mean2011,mean2012,mean2013,mean2014]
    # production
                mean2011=graphdata[(graphdata['Crop_Year']==2011) & (graphdata['State_Name']=='Bihar')]
                pmean2011=mean2011['Production'].mean()

                mean2012=graphdata[(graphdata['Crop_Year']==2012) & (graphdata['State_Name']=='Bihar') ]
                pmean2012=mean2012['Production'].mean()

                mean2014=graphdata[(graphdata['Crop_Year']==2014) & (graphdata['State_Name']=='Bihar') ]
                pmean2014=mean2014['Production'].mean()

                mean2013=graphdata[(graphdata['Crop_Year']==2013) & (graphdata['State_Name']=='Bihar')]
                pmean2013=mean2013['Production'].mean()

                mean2010=graphdata[(graphdata['Crop_Year']==2010) & (graphdata['State_Name']=='Bihar')]
                pmean2010=mean2010['Production'].mean()

                mean2009=graphdata[(graphdata['Crop_Year']==2009) & (graphdata['State_Name']=='Bihar')]
                pmean2009=mean2009['Production'].mean()
            
                pmeanlist=[pmean2009,pmean2010,pmean2011,pmean2012,pmean2013,pmean2014]
                yearlist=graphdata['Crop_Year'].unique()       
                return(cppred,sppred,productionpred,meanlist,yearlist,pmeanlist)
            
            elif (name=='gujarat'):
                gujdata=data1[data1['State_Name']=='Gujarat']
                cols=gujdata.columns.tolist()
                independentfeatures=sorted(list(set(cols)-set(['Production','Sunshine','Relative Humidity','Cost Price','Unnamed: 0','Crop','District_Name','Selling Price','Season','State_Name'])))
                y=gujdata['Production'].values
                x=gujdata[independentfeatures].values
                
                x=np.append(x,[[Area,2020,125,20.59]],axis=0)
                
                x_std = preprocessing.scale(x)
                
                userinput=x_std[81,:]
                x_std=np.delete(x_std,81,axis=0)
                
                train_x,test_x,train_y,test_y=train_test_split(x_std,y,test_size=0.1,random_state=0)
                
                poly=PolynomialFeatures(degree = 1)
                x_poly=poly.fit_transform(train_x)
                
                poly.fit(x_poly,train_y)
                linear2=LinearRegression()
                linear2.fit(x_poly,train_y)
                
                pred=linear2.predict(poly.fit_transform(test_x))            
                print(r2_score(test_y,pred))
                productionpred=linear2.predict(poly.fit_transform(userinput.reshape(1,-1)))
                
                x=gujdata['Production'].values
                y=gujdata['Cost Price'].values
                
                x=np.append(x,productionpred,axis=0)
                
                x_std = preprocessing.scale(x)
                
                userinput=x_std[81]
                x_std=np.delete(x_std,81,axis=0)
                
                linear2=LinearRegression()
                linear2.fit(x_std.reshape(-1,1),y.reshape(-1,1))
                
                cppred=linear2.predict(userinput.reshape(1,-1))
                
                data2=pd.read_excel('C:\\Users\\aayus\\project\\cotton_prediction\\costvssp.xlsx')
            
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
                    
                graphdata=pd.read_csv('C:\\Users\\aayus\\project\\cotton_prediction\\maizedataset.csv',encoding='cp1252')
                mean2011=graphdata[(graphdata['Crop_Year']==2011) & (graphdata['State_Name']=='Gujarat')]
                mean2011=mean2011['Cost Price'].mean()

                mean2012=graphdata[(graphdata['Crop_Year']==2012) & (graphdata['State_Name']=='Gujarat')]
                mean2012=mean2012['Cost Price'].mean()

                # mean2014=graphdata[(graphdata['Crop_Year']==2014) & (graphdata['State_Name']=='Gujarat')]
                # mean2014=mean2014['Cost Price'].mean()

                # mean2013=graphdata[(graphdata['Crop_Year']==2013) & (graphdata['State_Name']=='Gujarat')]
                # mean2013=mean2013['Cost Price'].mean()

                mean2010=graphdata[(graphdata['Crop_Year']==2010) & (graphdata['State_Name']=='Gujarat')]
                mean2010=mean2010['Cost Price'].mean()

                mean2009=graphdata[(graphdata['Crop_Year']==2009) & (graphdata['State_Name']=='Gujarat')]
                mean2009=mean2009['Cost Price'].mean()

                meanlist=[mean2009,mean2010,mean2011,mean2012]
                print(meanlist)
                # production

                mean2011=graphdata[(graphdata['Crop_Year']==2011) & (graphdata['State_Name']=='Gujarat')]
                pmean2011=mean2011['Production'].mean()

                mean2012=graphdata[(graphdata['Crop_Year']==2012) & (graphdata['State_Name']=='Gujarat')]
                pmean2012=mean2012['Production'].mean()

                # mean2014=graphdata[(graphdata['Crop_Year']==2014) & (graphdata['State_Name']=='Gujarat')]
                # pmean2014=mean2014['Production'].mean()

                # mean2013=graphdata[(graphdata['Crop_Year']==2013) & (graphdata['State_Name']=='Gujarat')]
                # pmean2013=mean2013['Production'].mean()

                mean2010=graphdata[(graphdata['Crop_Year']==2010) & (graphdata['State_Name']=='Gujarat')]
                pmean2010=mean2010['Production'].mean()

                mean2009=graphdata[(graphdata['Crop_Year']==2009) & (graphdata['State_Name']=='Gujarat')]
                pmean2009=mean2009['Production'].mean()

                pmeanlist=[pmean2009,pmean2010,pmean2011,pmean2012,]
                yearlist=graphdata['Crop_Year'].unique()    
                return(sppred,cppred,productionpred,meanlist,yearlist,pmeanlist)
                
            
            elif name=='uttarpradesh':           
                updata=data1[data1['State_Name']=='Uttar Pradesh']
                cols=updata.columns.tolist()
                userinput=np.array([Area,2020,125,20.59])
            
                independentfeatures=sorted(list(set(cols)-set(['Production','Sunshine','Relative Humidity','Cost Price','Unnamed: 0','Crop','District_Name','Selling Price','Season','State_Name'])))
                y=updata['Production'].values
                x=updata[independentfeatures].values
                train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.2,random_state=0)
                regressor=SVR(kernel='linear')
                regressor.fit(train_x,train_y)
                pred=regressor.predict(test_x)            
                print(r2_score(test_y,pred)) #87
                
                productionpred=regressor.predict(userinput.reshape(1,-1))
                
                x=updata['Production'].values
                y=updata['Cost Price'].values
                
                x=np.append(x,productionpred,axis=0)
                
                x_std = preprocessing.scale(x)
                
                userinput=x_std[589]
                x_std=np.delete(x_std,589,axis=0)
                
                linear2=LinearRegression()
                linear2.fit(x_std.reshape(-1,1),y.reshape(-1,1))
                
                cppred=linear2.predict(userinput.reshape(1,-1))
                
                data2=pd.read_excel('C:\\Users\\aayus\\project\\cotton_prediction\\costvssp.xlsx')
            
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
                
                if sb == 1:    
                    graphdata=pd.read_csv('C:\\Users\\aayus\\project\\cotton_prediction\\maizedataset.csv',encoding='cp1252')
                mean2011=graphdata[(graphdata['Crop_Year']==2011) & (graphdata['State_Name']=='Uttar Pradesh')]
                mean2011=mean2011['Cost Price'].mean()

                mean2012=graphdata[(graphdata['Crop_Year']==2012) & (graphdata['State_Name']=='Uttar Pradesh')]
                mean2012=mean2012['Cost Price'].mean()

                mean2014=graphdata[(graphdata['Crop_Year']==2014) & (graphdata['State_Name']=='Uttar Pradesh')]
                mean2014=mean2014['Cost Price'].mean()

                mean2013=graphdata[(graphdata['Crop_Year']==2013) & (graphdata['State_Name']=='Uttar Pradesh')]
                mean2013=mean2013['Cost Price'].mean()

                mean2010=graphdata[(graphdata['Crop_Year']==2010) & (graphdata['State_Name']=='Uttar Pradesh')]
                mean2010=mean2010['Cost Price'].mean()

                mean2009=graphdata[(graphdata['Crop_Year']==2009) & (graphdata['State_Name']=='Uttar Pradesh')]
                mean2009=mean2009['Cost Price'].mean()
                meanlist=[mean2009,mean2010,mean2011,mean2012,mean2013,mean2014]
                # production
                mean2011=graphdata[(graphdata['Crop_Year']==2011) & (graphdata['State_Name']=='Uttar Pradesh')]
                pmean2011=mean2011['Production'].mean()

                mean2012=graphdata[(graphdata['Crop_Year']==2012) & (graphdata['State_Name']=='Uttar Pradesh')]
                pmean2012=mean2012['Production'].mean()

                mean2014=graphdata[(graphdata['Crop_Year']==2014) & (graphdata['State_Name']=='Uttar Pradesh')]
                pmean2014=mean2014['Production'].mean()

                mean2013=graphdata[(graphdata['Crop_Year']==2013) & (graphdata['State_Name']=='Uttar Pradesh')]
                pmean2013=mean2013['Production'].mean()

                mean2010=graphdata[(graphdata['Crop_Year']==2010) & (graphdata['State_Name']=='Uttar Pradesh')]
                pmean2010=mean2010['Production'].mean()

                mean2009=graphdata[(graphdata['Crop_Year']==2009) & (graphdata['State_Name']=='Uttar Pradesh')]
                pmean2009=mean2009['Production'].mean()

            
                pmeanlist=[pmean2009,pmean2010,pmean2011,pmean2012,pmean2013,pmean2014]
                yearlist=graphdata['Crop_Year'].unique()     
                return(cppred,sppred,productionpred,meanlist,yearlist,pmeanlist)
            
            elif name=='tamilnadu':
                #NO SVR
                tndata=data1[data1['State_Name']=='Tamil Nadu']
                cols=tndata.columns.tolist()
                
                independentfeatures=sorted(list(set(cols)-set(['Production','Sunshine','Relative Humidity','Cost Price','Unnamed: 0','Crop','District_Name','Selling Price','Season','State_Name'])))
                y=tndata['Production'].values
                x=tndata[independentfeatures].values
                
                x=np.append(x,[[Area,2020,125,20.59]],axis=0)
                
                x_std = preprocessing.scale(x)
                
                userinput=x_std[127,:]
                x_std=np.delete(x_std,127,axis=0)
                
                train_x,test_x,train_y,test_y=train_test_split(x_std,y,test_size=0.1,random_state=0)
                
                poly=PolynomialFeatures(degree = 1)
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
                
                userinput=x_std[127]
                x_std=np.delete(x_std,127,axis=0)
                
                linear2=LinearRegression()
                linear2.fit(x_std.reshape(-1,1),y.reshape(-1,1))
                
                cppred=linear2.predict(userinput.reshape(1,-1))
                
                data2=pd.read_excel('C:\\Users\\aayus\\project\\cotton_prediction\\costvssp.xlsx')
            
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
                
                if sb == 1:    
                    graphdata=pd.read_csv('C:\\Users\\aayus\\project\\cotton_prediction\\maizedataset.csv',encoding='cp1252')
                mean2011=graphdata[(graphdata['Crop_Year']==2011) & (graphdata['State_Name']=='Tamil Nadu')]
                mean2011=mean2011['Cost Price'].mean()

                mean2012=graphdata[(graphdata['Crop_Year']==2012) & (graphdata['State_Name']=='Tamil Nadu')]
                mean2012=mean2012['Cost Price'].mean()

                # mean2014=graphdata[(graphdata['Crop_Year']==2014) & (graphdata['State_Name']=='Tamil Nadu')]
                # mean2014=mean2014['Cost Price'].mean()

                # mean2013=graphdata[(graphdata['Crop_Year']==2013) & (graphdata['State_Name']=='Tamil Nadu')]
                # mean2013=mean2013['Cost Price'].mean()

                mean2010=graphdata[(graphdata['Crop_Year']==2010) & (graphdata['State_Name']=='Tamil Nadu')]
                mean2010=mean2010['Cost Price'].mean()

                mean2009=graphdata[(graphdata['Crop_Year']==2009) & (graphdata['State_Name']=='Tamil Nadu')]
                mean2009=mean2009['Cost Price'].mean()
                meanlist=[mean2009,mean2010,mean2011,mean2012]
            # production
                mean2011=graphdata[(graphdata['Crop_Year']==2011) & (graphdata['State_Name']=='Tamil Nadu')]
                pmean2011=mean2011['Production'].mean()

                mean2012=graphdata[(graphdata['Crop_Year']==2012) & (graphdata['State_Name']=='Tamil Nadu')]
                pmean2012=mean2012['Production'].mean()

                # mean2014=graphdata[(graphdata['Crop_Year']==2014) & (graphdata['State_Name']=='Tamil Nadu')]
                # pmean2014=mean2014['Production'].mean()

                # mean2013=graphdata[(graphdata['Crop_Year']==2013) & (graphdata['State_Name']=='Tamil Nadu')]
                # pmean2013=mean2013['Production'].mean()

                mean2010=graphdata[(graphdata['Crop_Year']==2010) & (graphdata['State_Name']=='Tamil Nadu')]
                pmean2010=mean2010['Production'].mean()

                mean2009=graphdata[(graphdata['Crop_Year']==2009) & (graphdata['State_Name']=='Tamil Nadu')]
                pmean2009=mean2009['Production'].mean()

                
                pmeanlist=[pmean2009,pmean2010,pmean2011,pmean2012]
                yearlist=graphdata['Crop_Year'].unique()      
                return(cppred,sppred,productionpred,meanlist,yearlist,pmeanlist)
            
            elif name=='karnataka':
                
                kardata=data1[data1['State_Name']=='Karnataka']
                cols=kardata.columns.tolist()
                userinput=np.array([Area,2020,125,20.59])

                independentfeatures=sorted(list(set(cols)-set(['Production','Sunshine','Relative Humidity','Cost Price','Unnamed: 0','Crop','District_Name','Selling Price','Season','State_Name'])))
                y=kardata['Production'].values
                x=kardata[independentfeatures].values
                train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.2,random_state=0)
                regressor=SVR(kernel='linear')
                regressor.fit(train_x,train_y)
                pred=regressor.predict(test_x)            
                print(r2_score(test_y,pred)) #87
                
                productionpred=regressor.predict(userinput.reshape(1,-1))
                
                x=kardata['Production'].values
                y=kardata['Cost Price'].values
                
                x=np.append(x,productionpred,axis=0)
                
                x_std = preprocessing.scale(x)
                
                userinput=x_std[238]
                x_std=np.delete(x_std,238,axis=0)
                
                linear2=LinearRegression()
                linear2.fit(x_std.reshape(-1,1),y.reshape(-1,1))
                
                cppred=linear2.predict(userinput.reshape(1,-1))
                
                data2=pd.read_excel('C:\\Users\\aayus\\project\\cotton_prediction\\costvssp.xlsx')
            
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
                
                if sb == 1:    
                    graphdata=pd.read_csv('C:\\Users\\aayus\\project\\cotton_prediction\\maizedataset.csv',encoding='cp1252')
                mean2011=graphdata[(graphdata['Crop_Year']==2011)  & (graphdata['State_Name']=='Karnataka')]
                mean2011=mean2011['Cost Price'].mean()

                mean2012=graphdata[(graphdata['Crop_Year']==2012)  & (graphdata['State_Name']=='Karnataka')]
                mean2012=mean2012['Cost Price'].mean()

                mean2014=graphdata[(graphdata['Crop_Year']==2014)  & (graphdata['State_Name']=='Karnataka')]
                mean2014=mean2014['Cost Price'].mean()

                mean2013=graphdata[(graphdata['Crop_Year']==2013)  & (graphdata['State_Name']=='Karnataka')]
                mean2013=mean2013['Cost Price'].mean()

                mean2010=graphdata[(graphdata['Crop_Year']==2010)  & (graphdata['State_Name']=='Karnataka')]
                mean2010=mean2010['Cost Price'].mean()

                mean2009=graphdata[(graphdata['Crop_Year']==2009)  & (graphdata['State_Name']=='Karnataka')]
                mean2009=mean2009['Cost Price'].mean()
                meanlist=[mean2009,mean2010,mean2011,mean2012,mean2013,mean2014]
            # production
                mean2011=graphdata[(graphdata['Crop_Year']==2011 ) & (graphdata['State_Name']=='Karnataka')]
                pmean2011=mean2011['Production'].mean()

                mean2012=graphdata[(graphdata['Crop_Year']==2012 ) & (graphdata['State_Name']=='Karnataka')]
                pmean2012=mean2012['Production'].mean()

                mean2014=graphdata[(graphdata['Crop_Year']==2014)  & (graphdata['State_Name']=='Karnataka')]
                pmean2014=mean2014['Production'].mean()

                mean2013=graphdata[(graphdata['Crop_Year']==2013)  & (graphdata['State_Name']=='Karnataka')]
                pmean2013=mean2013['Production'].mean()

                mean2010=graphdata[(graphdata['Crop_Year']==2010)  & (graphdata['State_Name']=='Karnataka')]
                pmean2010=mean2010['Production'].mean()

                mean2009=graphdata[(graphdata['Crop_Year']==2009)  & (graphdata['State_Name']=='Karnataka')]
                pmean2009=mean2009['Production'].mean()

                
                pmeanlist=[pmean2009,pmean2010,pmean2011,pmean2012,pmean2013,pmean2014]
                yearlist=graphdata['Crop_Year'].unique()      
                return(cppred,sppred,productionpred,meanlist,yearlist,pmeanlist)
        
       
def index(request):
  
    # return render(request,'regionselect.html',{'forminput': forminput}) 
    return render(request,'Aboutus.html')

def userprofile(request):
    return render(request, 'User.html')

def predictions(request):

    data1=pd.read_csv('C:/Users/aayus/Desktop/MahaData.csv')
    data2=data1.copy(deep=True)
    # data2.set_index('Year',inplace=True)
    year = data2['Year']

    return render(request, 'Predictions.html')

def regionselect(request):
    if request.method == 'GET':
        forminput=CommodityForm()
        # return render(request,'regionselect.html',{'forminput': forminput})
    return render(request,'regionselectcom.html',{'forminput': forminput})

def areadata(request):
    if request.method == "POST":
        forminput2 = CommodityForm(request.POST)
        if forminput2.is_valid():
            com = forminput2.cleaned_data['commodity']

            if com == "rice":
                forminput = RiceForm()
                return render(request,'regionselect.html',{'forminput': forminput, 'iny':2})

            if com == "maize":
                forminput = MaizeForm()
                return render(request,'regionselect.html',{'forminput': forminput, 'iny':1})

            if com == "cotton":
                forminput = CottonForm()
                return render(request,'regionselect.html',{'forminput': forminput, 'iny':3})        


def formdata(request, sb):

    if request.method=="POST": 
        
        if sb == 1:
            forminput=MaizeForm(request.POST)
            text = 'maize'

        if sb == 2:
            forminput=RiceForm(request.POST)
            text = 'rice'
        
        if sb == 3:
            forminput=CottonForm(request.POST) 
            text = 'cotton'       
        
        if forminput.is_valid():
                                                                                                            
            Area1=forminput.cleaned_data['area']
            Region=forminput.cleaned_data['region']
            if sb == 1:
                sellp,costp,pred,meanlist,yearlist,pmeanlist = calc12(Region,Area1,sb)

            if sb == 2:
                if Region == 'madhyapradesh':    
                    sellp,costp,pred,meanlist,yearlist,pmeanlist = calc3(Area1) 
                        
                if Region == 'orissa':
                    sellp,costp,pred,meanlist,yearlist,pmeanlist = calc4(Area1,1) 
                if Region == 'westbengal':
                    sellp,costp,pred,meanlist,yearlist,pmeanlist = calc4(Area1,2)   
                if Region == 'assam':
                    sellp,costp,pred,meanlist,yearlist,pmeanlist = calc4(Area1,3)  
                if Region == 'uttarpradesh':
                    sellp,costp,pred,meanlist,yearlist,pmeanlist = calc4(Area1,4)           
            # print(sellp)
            if sb == 3:
                if Region == 'karnataka' or Region == 'tamilnadu':
                    sellp,costp,pred,meanlist,yearlist,pmeanlist = calc5(Region,Area1) 
                if Region == 'maharashtra':
                    sellp,costp,pred,meanlist,yearlist,pmeanlist = calc4(Area1,1)     
            
            sellp=float(sellp)
            costp=float(costp[0])
            pred=int(pred[0])
            
            costp=round(costp, 2)
            sellp=round(sellp, 2)
            
            hist = Prediction.objects.all().last()
            i = hist.idn
            i=i+1
            da=datetime.now()
            print(meanlist)
            Prediction.objects.create(idn=i,area =Area1, region=Region, commod=text, daten=da)
    return render(request,'Predictions.html',{'propred':pred,'costp':costp,'sellp':sellp,
                                              'meanlist':meanlist,'yearlist':yearlist,'pmeanlist':pmeanlist})
 
def history(request):
    data = Prediction.objects.all()
    return render(request,'history.html',{'data':data})

def hisdata(request,hs):
    print(hs)
    data = Prediction.objects.get(idn=hs)
    print(data)
    com= data.commod
    ar= data.area
   
    reg = data.region
    if com == 'maize':
        sb = 1
    if com == 'rice':
        sb = 2
    if com == 'cotton':
        sb = 3        
    
    if sb == 1:
        sellp,costp,pred,meanlist,yearlist,pmeanlist = calc12(reg,ar,sb)
    if sb == 2:
        if reg == 'madhyapradesh':    
            sellp,costp,pred,meanlist,yearlist,pmeanlist = calc3(ar) 
                
        if reg == 'orissa':
            sellp,costp,pred,meanlist,yearlist,pmeanlist = calc4(ar,1) 
        if reg == 'westbengal':
            sellp,costp,pred,meanlist,yearlist,pmeanlist = calc4(ar,2)   
        if reg == 'assam':
            sellp,costp,pred,meanlist,yearlist,pmeanlist = calc4(ar,3)  
        if reg == 'uttarpradesh':
            sellp,costp,pred,meanlist,yearlist,pmeanlist = calc4(ar,4)           
            # print(sellp)
    if sb == 3:
        if reg == 'karnataka' or reg == 'tamilnadu':
            sellp,costp,pred,meanlist,yearlist,pmeanlist = calc5(reg,ar) 

    # sellp=sellp[0][0]
    sellp=float(sellp)
    costp=float(costp[0])
    pred=int(pred[0])
            
    costp=round(costp, 2)
    sellp=round(sellp, 2)
                     
    return render(request,'Predictions.html',{'propred':pred,'costp':costp,'sellp':sellp,
                                              'meanlist':meanlist,'yearlist':yearlist,'pmeanlist':pmeanlist})

def hisdel(request,hs):
    db = Prediction.objects.get(idn=hs)
    db.delete()
    
    data = Prediction.objects.all()
    return render(request,'history.html',{'data':data})




class ClubChartView(TemplateView):
    template_name='Predictions.html'

    def get_context_data(self, **kwargs):

       
        context= super().get_context_data(**kwargs)
        # context['qs']= meanlist  #database extraction
        return context