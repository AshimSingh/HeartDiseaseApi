from django.db import models

class PredictionResult(models.Model):
    Age = models.IntegerField(default=0)
    Sex = models.IntegerField(default=0)
    chest_pain_type = models.IntegerField(default=0)
    BP = models.IntegerField(default=0)
    Cholesterol = models.IntegerField(default=0)
    Prediction = models.IntegerField(default=0)
    