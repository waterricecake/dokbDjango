from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('dapi/', include('dapi.urls')),
    path('admin/', admin.site.urls),
    ]
