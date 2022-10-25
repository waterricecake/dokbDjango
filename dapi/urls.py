
from django.urls import path
from . import views

app_name="dapi"

urlpatterns = [
    path('certification/', views.certification, name='certification'),
    # path('send/', views.send, name='send'),
    path('predict/', views.pred, name='predict'),
]