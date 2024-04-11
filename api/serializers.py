from rest_framework import serializers

from .models import PredictionResult


        
class PredictionResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = PredictionResult
        fields =("Age" ,"Sex","Chest pain type","BP","Cholesterol","Prediction ")