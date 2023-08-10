import os,sys
from Insurance.exception import InsuranceException
from Insurance.logger import logging
from datetime import datetime
from Insurance.utils import read_yaml_file
from Insurance.constant import *


FILE_NAME = "insurance.csv"


config_data=read_yaml_file(CONFIG_FILE_PATH)

class TrainingPipelineConfig:
    
    def __init__(self):
        try:
            self.artifact_dir = os.path.join(os.getcwd(),"artifact")
            
            
        except Exception  as e:
            raise InsuranceException(e,sys)    


class DataIngestionConfig:
    
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        try:
            data_ingestion_key=config_data[DATA_INGESTION_CONFIG_KEY]
            
            self.database_name=data_ingestion_key[DATA_INGESTION_DATABASE_NAME]
            self.collection_name=data_ingestion_key[DATA_INGESTION_COLLECTION_NAME]
            
            
            self.root_directory=os.path.join(os.getcwd(),'Data')
            self.data_ingestion_dir = os.path.join(training_pipeline_config.artifact_dir ,data_ingestion_key[DATA_INGESTION_ARTIFACT_DIR])
            self.raw_data_dir = os.path.join(self.data_ingestion_dir,data_ingestion_key[DATA_INGESTION_RAW_DATA_DIR_KEY])
            self.ingested_data_dir=os.path.join(self.raw_data_dir,data_ingestion_key[DATA_INGESTION_INGESTED_DIR_NAME_KEY])
            self.train_file_path = os.path.join(self.ingested_data_dir,data_ingestion_key[DATA_INGESTION_TRAIN_DIR_KEY])
            self.test_file_path = os.path.join(self.ingested_data_dir,data_ingestion_key[DATA_INGESTION_TEST_DIR_KEY])
            self.test_size = 0.2
        except Exception  as e:
            raise InsuranceException(e,sys)      



            
# Convert data into dict
    def to_dict(self,)->dict:
        try:
            return self.__dict__
        except Exception  as e:
            raise InsuranceException(e,sys)          

class DataValidationConfig:
    
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        
        data_validation_key=config_data[DATA_VALIDATION_CONFIG_KEY]
        
        self.data_validation_dir = os.path.join(training_pipeline_config.artifact_dir ,data_validation_key[DATA_VALIDATION_ARTIFACT_DIR])
        self.validated_dir=os.path.join(training_pipeline_config.artifact_dir,data_validation_key[DATA_VALIDATION_VALID_DATASET])
        self.validated_train_path=os.path.join(self.data_validation_dir,data_validation_key[DATA_VALIDATION_TRAIN_FILE])
        self.validated_test_path=os.path.join(self.data_validation_dir,data_validation_key[DATA_VALIDATION_TEST_FILE])
        self.schema_file_path=SCHEMA_FILE_PATH
        



class DataTransformationConfig:
    
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        
        
        data_transformation_key=config_data[DATA_TRANSFORMATION_CONFIG_KEY]
        
        self.data_transformation_dir = os.path.join(training_pipeline_config.artifact_dir , data_transformation_key[DATA_TRANSFORMATION])
        self.transformation_dir = os.path.join(self.data_transformation_dir,data_transformation_key[DATA_TRANSFORMATION_DIR_NAME_KEY])
        self.transformed_train_path =  os.path.join(self.transformation_dir,data_transformation_key[DATA_TRANSFORMATION_TRAIN_DIR_NAME_KEY],'X_train.npz')
        self.transformed_test_path =os.path.join(self.transformation_dir,data_transformation_key[DATA_TRANSFORMATION_TEST_DIR_NAME_KEY],'X_test.npz')
        self.target_train_file_path=os.path.join(self.transformation_dir,data_transformation_key[DATA_TRANSFORMATION_TRAIN_DIR_NAME_KEY],'train_target.npz')
        self.target_test_file_path=os.path.join(self.transformation_dir,data_transformation_key[DATA_TRANSFORMATION_TEST_DIR_NAME_KEY],'test_target.npz')
        self.preprocessed_dir = os.path.join(self.data_transformation_dir,data_transformation_key[DATA_TRANSFORMATION_PREPROCESSING_DIR_KEY])
        self.preprocessed_object = os.path.join(self.preprocessed_dir,data_transformation_key[DATA_TRANSFORMATION_PREPROCESSING_FILE_NAME_KEY])
        
        

class Hyperparamter_optimize:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        
        self.param_data=config_data[PARAMTERS_OPTIMIZATION_KEY]
        self.parameter_directory=os.path.join(training_pipeline_config.artifact_dir,self.param_data[PARAMTERS_OPTIMIZATION_DIR])
        self.param_file_path=os.path.join(self.parameter_directory,self.param_data[PARAMTERS_OPTIMIZATION_FILE_NAME])
        
        

class ModelTrainingConfig:
    
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        
        model_training_key=config_data[MODEL_TRAINING_CONFIG_KEY]
        
        self.model_training_dir = os.path.join(training_pipeline_config.artifact_dir ,model_training_key[MODEL_TRAINER_ARTIFACT_DIR])
        self.model_object = os.path.join(self.model_training_dir,model_training_key[MODEL_TRAINER_OBJECT])
        self.model_report =  os.path.join(self.model_training_dir,model_training_key[MODEL_REPORT_FILE])
        
class ModelEvalConfig:
    
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        
        
        model_eval_config_key=config_data[MODEL_EVAL_CONFIG_KEY]        
        
        self.model_eval_directory=os.path.join(training_pipeline_config.artifact_dir ,model_eval_config_key[MODEL_EVALUATION_DIRECTORY])
        self.model_eval_report=os.path.join(self.model_eval_directory,model_eval_config_key[MODEL_REPORT])
        

class SavedModelConfig:
    
    def __init__(self):
        saved_model_config_key=config_data[SAVED_MODEL_CONFIG_KEY]
        
        ROOT_DIR=os.getcwd()
        self.saved_model_dir=os.path.join(ROOT_DIR,saved_model_config_key[SAVED_MODEL_DIR])
        self.saved_model_file_path=os.path.join(self.saved_model_dir,saved_model_config_key[SAVED_MODEL_OBJECT])
        self.saved_model_report_path=os.path.join(self.saved_model_dir,saved_model_config_key[SAVED_MODEL_REPORT])


