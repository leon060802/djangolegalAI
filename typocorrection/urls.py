from django.urls import path
from . import views
urlpatterns = [
    path('', views.upload_and_process_pdf, name="upload_and_process_pdf"),
]