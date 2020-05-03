from django.db import models
from datetime import datetime
from django.utils import timezone
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
    commod = models.CharField(max_length=100,null = True)
    idn = models.IntegerField(default=0)
    daten = models.DateField(default = timezone.now)