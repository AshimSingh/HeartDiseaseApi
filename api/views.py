
from rest_framework.response import Response
from rest_framework import status
from rest_framework.views import APIView
import pandas as pd
from .ID3algorithm import build_decision_tree, classify_instance
from .cleandata import cleandata
from django.http import JsonResponse
# from .sklearnPredictionAlgorithm import makeDecisionTree,predict
# from .outlookprediction import build_decision_tree, classify_instance
     
class PredictionRead(APIView):
    def post(self,request):
        test_instance = request.data  
        data = pd.read_csv(r'D:\Api\heartdiseaseapi\api\Heart_Disease_Prediction.csv')
        df = cleandata(data)
        print(df)
        decision_tree = build_decision_tree(df)
        prediction = classify_instance(test_instance, decision_tree)
        return Response({"prediction ":prediction})
        # return Response({"hello":"ashim"})
    def get(self,request):
        data = {"message": "Hello welcome to our project"}
        return JsonResponse(data, status=status.HTTP_200_OK)

# class SklearnPrediction(APIView):

#     def post(self,request):
#         test_instance = request.data
#         data = pd.read_csv(r'F:\\HeartDiseasePrediction\\users\\Heart_Disease_Prediction.csv')
#         # df = cleandata(data)
#         decision_tree = makeDecisionTree(data)
#         newData = pd.DataFrame([test_instance])
#         predction = predict(data=newData,tree=decision_tree)
#         # print(predction,'my prediction')
#         print(newData.dtypes)
#         # prediction = classify_instance(test=test_instance,tree=decision_tree)
#         # return Response({"prediction ":prediction})
#         return Response({"hello":"ashim"})