from django.contrib import admin
from django.urls import path
from . import views

app_name = "predict"

urlpatterns = [
    path("init/", views.init, name='init'),
    path("index/", views.index, name='index'),
    path("home/", views.home, name = 'home'),
    path("predict/", views.predict_stock_action, name = 'predict'),
]
