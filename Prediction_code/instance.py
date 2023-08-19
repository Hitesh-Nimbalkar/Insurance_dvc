
import os
import logging
from Insurance.logger import logging
from Insurance.exception import InsuranceException
import pandas as pd
import pickle
from sklearn.pipeline import Pipeline
from Insurance.utils import read_yaml_file
from Insurance.entity.artifact_entity import ModelTrainerArtifact,DataTransformationArtifact
import sys 
from Insurance.utils import read_yaml_file,load_object


INSTANCE_PREDICTION="Instance_prediction"

transformer_file_path ="Preprocessed/preprocessor.pkl"
model_path ="Saved_model/model.pkl"






import pandas as pd
import joblib

# Load the preprocessor and machine learning model
preprocessor = load_object('Preprocessed_files/preprocessor.pkl')
model = load_object('Saved_model/model.pkl')


class instance_prediction_class:
    def __init__(self, age, 
                 sex, 
                 bmi,
                 children, 
                 smoker,region) -> None:
        self.age = age
        self.sex = sex
        self.bmi = bmi
        self.children = children
        self.smoker = smoker
        self.region = region
 
    def preprocess_input(self):
        # Create a DataFrame with the user input
        user_input = pd.DataFrame({
            'age': [self.age],
            'sex': [self.sex],
            'bmi': [self.bmi],
            'children': [self.children],
            'smoker': [self.smoker],
            'region': [self.region]
            
        })
        
        print(user_input)

        # Preprocess the user input using the preprocessor
        preprocessed_array = preprocessor.transform(user_input)
        
        print(preprocessed_array)
        return preprocessed_array

    def prediction(self, preprocessed_input):
        # Make a prediction using the pre-trained model
        predicted_expenses = model.predict(preprocessed_input)

        # Return the array of predicted prices
        return predicted_expenses

    def predict_expense(self):
        # Preprocess the input using the preprocessor
        preprocessed_array = self.preprocess_input()

        # Make a prediction using the pre-trained model
        predicted_expenses = self.prediction(preprocessed_array)

        # Round off the predicted shipment prices to two decimal places
        expense = [round(expense, 2) for expense in predicted_expenses]

        # Print the rounded predicted shipment prices
        
        print(f"The predicted Concrete Strength  is:  {expense[0]}")

        return expense[0]