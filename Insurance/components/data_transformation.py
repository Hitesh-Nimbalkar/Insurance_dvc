from Insurance.entity import artifact_entity,config_entity
from Insurance.exception import InsuranceException
from Insurance.logger import logging
import sys 
from sklearn.pipeline import Pipeline
import pandas as pd
from Insurance import utils
import numpy as np
from sklearn.preprocessing import LabelEncoder
from Insurance.constant import *
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class DataTransformation:


    def __init__(self,data_transformation_config:config_entity.DataTransformationConfig,
                    data_validation_artifact:artifact_entity.DataValidationArtifact):
        try:
            self.data_transformation_config=data_transformation_config
            self.data_validation_artifact=data_validation_artifact
            
            # Column labels 
            schema_data=utils.read_yaml_file(file_path=SCHEMA_FILE_PATH)
            
            self.numerical_columns=schema_data['numerical_columns']
            self.categorical_columns=schema_data['categorical_columns']
            
        except Exception as e:
            raise InsuranceException(e, sys)


   
    def get_data_transformer_object(self)->Pipeline: 
        try:

            # Create the data preprocessing pipeline
            pipeline = ColumnTransformer([
                ('categorical', OneHotEncoder(), self.categorical_columns),
                ('numeric', StandardScaler(), self.numerical_columns)
            ])

            return pipeline
        except Exception as e:
            raise InsuranceException(e, sys)


    def initiate_data_transformation(self,) -> artifact_entity.DataTransformationArtifact:
        try:
            train_df = pd.read_csv(self.data_validation_artifact.validated_train_path)
            test_df = pd.read_csv(self.data_validation_artifact.validated_test_path)
            
            TARGET_COLUMN='expenses'
            input_feature_train_df=train_df.drop(TARGET_COLUMN,axis=1)
            input_feature_test_df=test_df.drop(TARGET_COLUMN,axis=1)

            target_feature_train = train_df[TARGET_COLUMN].values
            target_feature_test = test_df[TARGET_COLUMN].values
            
            # Preprocessing 
            transformation_pipleine = self.get_data_transformer_object()
            X_train_preprocessed = transformation_pipleine.fit_transform(input_feature_train_df)

            # Apply the same transformations on X_test (without fitting again)
            X_test_preprocessed = transformation_pipleine.transform(input_feature_test_df)


            # Saving Numpy arrays 
            utils.save_numpy_array_data(file_path=self.data_transformation_config.transformed_train_path  ,
                                        array=X_train_preprocessed)

            utils.save_numpy_array_data(file_path=self.data_transformation_config.transformed_test_path ,
                                        array=X_test_preprocessed)

            # Save target Features 
            utils.save_numpy_array_data(file_path=self.data_transformation_config.target_train_file_path ,
                                        array=target_feature_train)
            utils.save_numpy_array_data(file_path=self.data_transformation_config.target_test_file_path ,
                                        array=target_feature_test)

            utils.save_object(file_path=self.data_transformation_config.preprocessed_object,
             obj=transformation_pipleine)
            
            current_directory=os.getcwd()
            object_directory=os.path.join(current_directory,'Preprocessed_files','preprocessor.pkl')
            utils.save_object(file_path=object_directory,
             obj=transformation_pipleine)


            data_transformation_artifact = artifact_entity.DataTransformationArtifact(
                transform_object_path=self.data_transformation_config.preprocessed_object,
                transformed_train_path = self.data_transformation_config.transformed_train_path,
                transformed_test_path = self.data_transformation_config.transformed_test_path,
                target_train=self.data_transformation_config.target_train_file_path,
                target_test=self.data_transformation_config.target_test_file_path
            )

            return data_transformation_artifact
        except Exception as e:
            raise InsuranceException(e, sys)