from django.contrib import admin
from django.urls import path, include
from . import views

urlpatterns = [
    # path('',views.regionselect,name='region'),
    # path('',views.aboutus, name='aboutus'),
    path('/user', views.userprofile, name= 'user'),
    path('', views.index, name='aboutus'),
    path('/prediction', views.predictions, name= 'prediction'),
    
    path('/regionselect', views.regionselect, name= 'regionselect'),
    path('/formdata', views.formdata, name='formdata'),
]