from django.urls import path

from . import views

urlpatterns = [
    path('input/', views.index,name='input'),
    #path('result/', views.result, name="result"),
    path('analyse/', views.analyse, name='analyse'),
]