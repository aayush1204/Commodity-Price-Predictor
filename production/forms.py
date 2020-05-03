from django import forms
# r=[('rajasthan','Rajasthan'),('maharashtra','Maharashtra'),('gujarat','Gujrat'),('punjab','Punjab'),('haryana','Haryana'),]
s=[('rice','Rice'),('cotton','Cotton'),('maize','Maize')]
r=[('assam','Assam'),('uttarpradesh','Uttar Pradesh'),('westbengal','West Bengal'),('orissa','Orissa'),('madhyapradesh','Madhya Pradesh'),]
m=[('andhrapradesh','Andhra Pradesh'),('bihar','Bihar'),('gujarat','Gujrat'),('karnataka','Karnataka'),('tamilnadu','Tamil Nadu'),('uttarpradesh','Uttar Pradesh'),]
c=[('tamilnadu','Tamil Nadu'),('maharashtra','Maharashtra'),('karnataka','Karnataka')]

# class AreaForm(forms.Form):
#     area = forms.CharField()
#     region = forms.ChoiceField(choices=r)

class CommodityForm(forms.Form):
    commodity = forms.ChoiceField(choices=s, widget = forms.Select(attrs ={'class' : 'form-control','background-color': '#27293d'})) 
class RiceForm(forms.Form):
    area = forms.CharField(widget=forms.TextInput (attrs={'class': 'form-control'} ))
    region = forms.ChoiceField(choices=r, widget = forms.Select(attrs ={'class' : 'form-control','background-color': '#27293d'}) )

    # class Meta:
    #     model = RiceForm

    #     widgets = {
    #         'region': forms.Select(attrs = {'class' : 'bootstrap-select'}),
    #     }

class MaizeForm(forms.Form):
    area = forms.CharField(widget=forms.TextInput (attrs={'class': 'form-control'} ))
    region = forms.ChoiceField(choices=m, widget = forms.Select(attrs ={'class' : 'form-control','background-color': '#27293d'}))

class CottonForm(forms.Form):
    area = forms.CharField(widget=forms.TextInput (attrs={'class': 'form-control'} ))
    region = forms.ChoiceField(choices=c, widget = forms.Select(attrs ={'class' : 'form-control','background-color': '#27293d'}))

