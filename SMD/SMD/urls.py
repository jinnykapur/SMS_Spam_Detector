# SMD/urls.py
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('', include('Spam.urls')),  # root is passed to Spam app
]
