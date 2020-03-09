from django.shortcuts import render
import requests
from .forms import AreaForm
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

# Create your views here.
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
        forminput=AreaForm()
        # return render(request,'regionselect.html',{'forminput': forminput})
    return render(request,'regionselect.html',{'forminput': forminput})

def formdata(request):
    
    if request.method=="POST":
        forminput=AreaForm(request.POST)
        if forminput.is_valid():
            
            Area=forminput.cleaned_data['area']
            Region=forminput.cleaned_data['region']
            data = Prediction.objects.create(area=Area,region=Region)
            data.save()
            #!/usr/bin/env python
        # coding: utf-8
        data1=pd.read_csv(r'C:\Users\aayus\project\cotton_prediction\MahaData.csv')

        data2=data1.copy(deep=True)

        data2['Area'].fillna(data2['Area'].median(),inplace=True)
        data2['Kg/Hectare'].fillna(data2['Kg/Hectare'].median(),inplace=True)
        data2['Rainfall(cm)'].fillna(data2['Rainfall(cm)'].mean(),inplace=True)
        y=data2['Kg/Hectare']
        x1=data2['Area']
        #data2['irrigation']=pd.to_numeric(data2['irrigation'],errors='coerce')
        #data2['Kg/Hectare']=pd.to_numeric(data2['Kg/Hectare'],errors='coerce')
        #data2['Area']=pd.to_numeric(data2['Area'],errors='coerce')

        data2.corr(method='pearson')
        data2.dropna(axis=1,inplace=True)

        columns_list=list(data2.columns)

        independentfeatures=sorted(list(set(columns_list)-set(['Kg/Hectare','Production'])))

        y=data2['Kg/Hectare'].values
        x=data2[independentfeatures].values


        sns.set(style="darkgrid")
        sns.regplot(x=data2['Rainfall(cm)'],y=data2['Kg/Hectare'])



        sns.set(style="white")
        sns.regplot(x=data2['Area'],y=data2['Kg/Hectare'])


        sns.set(style="ticks")
        sns.regplot(x=data2['Year'],y=data2['Kg/Hectare'])


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

        x2=np.array([Area,83,125,2535,20.59,2019]).reshape(1,6)

        prediction_val2=linear2.predict(poly.fit_transform(x2))
        print("Predicted value for 2019: ")
        print(prediction_val2) #KG/HECTARE


        data2[independentfeatures]


        columns_list=list(data2.columns)

        independentfeatures=sorted(list(set(columns_list)-set(['Production'])))

        y=data2['Production'].values.reshape(-1,1)
        x=data2['Kg/Hectare'].values.reshape(-1,1)


        sns.set(style="ticks")
        sns.regplot(x=data2['Kg/Hectare'],y=data2['Production'])



        train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)


        linear=LinearRegression()
        linear.fit(train_x,train_y)

        prediction_val1=linear.predict(test_x)

        print("Predicted values: ")
        print(prediction_val1)

        print(r2_score(test_y,prediction_val1))

        x2=np.array([prediction_val2])
        print(x2)

        prediction_val21=linear.predict(x2)
        print("Predicted value for 2019: ")
        print( prediction_val21)  #PRODUCTION
                
        data1=pd.read_csv(r'C:\Users\aayus\project\cotton_prediction\prodvscp.csv')
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


        prediction_cp=linear.predict([[5950]])
        print("Predicted value for 2019: ")
        print(prediction_cp)  #COST PRICE

        
        data1=pd.read_csv(r'C:\Users\aayus\project\cotton_prediction\Phase2Data.csv')

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

        x2=np.array([15,3000,prediction_cp,4000,2019]).reshape(1,5)

        prediction_sp=linear2.predict(poly.fit_transform(x2))
        print("Predicted value for 2019: ")
        print(prediction_sp)
       
        
    return render(request,'Predictions.html',{'pred2':prediction_val21[0][0],'pred1':prediction_cp[0][0],'pred3':prediction_sp[0],})

class ClubChartView(TemplateView):
    template_name='Predictions.html'

    def get_context_data(self, **kwargs):
        context= super().get_context_data(**kwargs)
        context['qs']= Prediction.objects.all()  #database extraction
        return context