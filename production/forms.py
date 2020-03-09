from django import forms
r=[('rajasthan','Rajasthan'),('maharashtra','Maharashtra'),('gujarat','Gujrat'),('punjab','Punjab'),('haryana','Haryana'),]
class AreaForm(forms.Form):
    area = forms.CharField()
    region = forms.ChoiceField(choices=r)