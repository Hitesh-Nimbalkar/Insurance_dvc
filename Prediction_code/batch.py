import os
import logging
from Insurance.logger import logging
from Insurance.exception import InsuranceException
import pandas as pd
import pickle
from sklearn.pipeline import Pipeline
from Insurance.utils import read_yaml_file,load_object
import sys 
import pymongo
import json
from Insurance.constant import *
from Insurance.constant import *
import urllib
import yaml
import numpy as np



BATCH_PREDICTION = 'batch_prediction/prediction'



class batch_prediction:
    def __init__(self,input_file_path, 
                 model_file_path, 
                 transformer_file_path
                 ) -> None:
        
        self.input_file_path = input_file_path
        self.model_file_path = model_file_path
        self.transformer_file_path = transformer_file_path

        
    
    def start_batch_prediction(self):
        try:
            logging.info("Loading the preprocessor pipeline")
            
            # Load the data transformation pipeline
            with open(self.transformer_file_path, 'rb') as f:
                preprocessor = pickle.load(f)

            logging.info(f"Preprocessor  Object acessed :{self.transformer_file_path}")
            
            # Load the model separately
            model =load_object(file_path=self.model_file_path)

            logging.info(f"Model File Path: {self.model_file_path}")

            # Read the input file
            df = pd.read_csv(self.input_file_path)


            df=df.drop('expenses', axis=1)       
                
                
            logging.info(f"Columns before transformation: {', '.join(f'{col}: {df[col].dtype}' for col in df.columns)}")
            # Transform the feature-engineered data using the preprocessor
            transformed_data = preprocessor.transform(df)
            logging.info(f"Transformed Data Shape: {transformed_data.shape}")
            predictions = model.predict(transformed_data)
            logging.info(f"Predictions done: {predictions}")

            # Round the values in the predictions array to the nearest integer
            rounded_predictions = [round(prediction) for prediction in predictions]

            # Create a DataFrame from the rounded_predictions array
            df_predictions = pd.DataFrame(rounded_predictions, columns=['expenses'])
            # Save the predictions to a CSV file
            BATCH_PREDICTION_PATH = BATCH_PREDICTION  # Specify the desired path for saving the CSV file
            os.makedirs(BATCH_PREDICTION_PATH, exist_ok=True)
            csv_path = os.path.join(BATCH_PREDICTION_PATH,'predictions.csv')
            df_predictions.to_csv(csv_path, index=False)
            logging.info(f"Batch predictions saved to '{csv_path}'.")

        except Exception as e:
            InsuranceException(e,sys) 

