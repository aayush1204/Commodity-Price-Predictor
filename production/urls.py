from django.contrib import admin
from django.urls import path, include
from . import views

urlpatterns = [
    # path('',views.regionselect,name='region'),
    # path('',views.aboutus, name='aboutus'),
    path('/user', views.userprofile, name= 'user'),
    path('', views.index, name='aboutus'),
    path('/prediction', views.predictions, name= 'prediction'),
    path('/history', views.history, name='history'),
    path('/regionselect', views.regionselect, name= 'regionselect'),
    path('/formdata', views.formdata, name='formdata'),
    path('/formdata/<int:sb>', views.formdata, name='formdata'),
    path('/areadata', views.areadata, name='areadata'),
    path('/hisdata/<int:hs>',views.hisdata, name='hisdata'),
    path('/hisdel/<int:hs>',views.hisdel, name='hisdel'),
]
