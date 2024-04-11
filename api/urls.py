from django.urls import path
from .views import PredictionRead


urlpatterns = [
    path('',PredictionRead.as_view(),name='prediction'),
    # path('sklearnpredict/',SklearnPrediction.as_view(),name='sklearnpredict')
]