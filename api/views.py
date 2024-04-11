
from rest_framework.response import Response
from rest_framework import status
from rest_framework.views import APIView
import pandas as pd
from .ID3algorithm import build_decision_tree, classify_instance
from .cleandata import cleandata
from django.http import JsonResponse
from .models import PredictionResult
from .serializers import PredictionResultSerializer
from .sklearnPredictionAlgorithm import makeDecisionTree,predict
# from .outlookprediction import build_decision_tree, classify_instance
     
class PredictionRead(APIView):
    def post(self,request):
        test_instance = request.data  
        data = pd.read_csv(r'D:\Api\heartdiseaseapi\api\Heart_Disease_Prediction.csv')
        df = cleandata(data)
        decision_tree = build_decision_tree(df)
        prediction = classify_instance(test_instance, decision_tree)
        pred_data = {**test_instance, 'prediction': prediction}
        serializer = PredictionResultSerializer(data=pred_data)
        # print(serializer.is_valid())
        # print(serializer.errors)
        if serializer.is_valid():
            # print('valid bebs')
            serializer.save()  # Save the serializer and get the instance
        # print(serializer,'hey ashimmmmmmmm')
        # print(pred_data)
        return Response(serializer.data, status=status.HTTP_200_OK)
       

    def get(self,request):
        my_pred_data = PredictionResult.objects.all()
        serializer  = PredictionResultSerializer(my_pred_data,many=True)
        # print(serializer.data)
        # data = serializer.data
        return Response({"data":serializer.data}, status=status.HTTP_200_OK)

class SklearnPrediction(APIView):

    def post(self,request):
        test_instance = request.data
        data = pd.read_csv(r'D:\Api\heartdiseaseapi\api\Heart_Disease_Prediction.csv')
        df = cleandata(data)
        decision_tree = makeDecisionTree(df)
        # print(decision_tree,'this decision tree')
        # newData = pd.DataFrame([test_instance])
        # predction = predict(data=newData,tree=decision_tree)
        
        # print(newData.dtypes)
        prediction = classify_instance(test=test_instance,tree=decision_tree)
        print(prediction)
        # return Response({"prediction ":prediction})
        return Response({"hello":"ashim"})