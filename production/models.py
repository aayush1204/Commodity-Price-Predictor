from django.db import models

# Create your models here.

# class Chart(models.Model):
#     year= models.IntegerField()
#     area = models.FloatField()
#     production=
#     rainfall
#     temp
#     sunshine
#     humidity
#     kg

class Prediction(models.Model):
    area = models.CharField(max_length=100)
    region = models.CharField(max_length=100)