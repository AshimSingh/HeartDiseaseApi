import pandas as py
import numpy as np


def cleandata(data):
    # cleaned_data = data.drop(['ChestPainType','RestingECG','MaxHR','ExerciseAngina','ST_Slope'],axis=1)
    cleaned_data = data.drop(['ST depression','Slope of ST','Number of vessels fluro','Thallium','FBS over 120','EKG results','Max HR','Exercise angina'],axis=1)
    return cleaned_data