from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name="home"),        # HOME PAGE
    path("upload/", views.upload_audio, name="upload"),

]
