from rest_framework import serializers

from .models import PredictionResult


        
class PredictionResultSerializer(serializers.ModelSerializer):
    # chest_pain_type = serializers.CharField(source='Chest pain type')
    prediction = serializers.IntegerField(source='Prediction')

    class Meta:
        model = PredictionResult
        fields = ['Age', 'Sex', 'BP','chest_pain_type', 'Cholesterol', 'prediction']