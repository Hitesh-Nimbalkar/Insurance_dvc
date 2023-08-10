
import os

FILENAME='insurance.csv'

# Schema File path
ROOT_DIR=os.getcwd()
CONFIG_DIR='config'
SCHEMA_FILE='schema.yaml'
SCHEMA_FILE_PATH=os.path.join(ROOT_DIR,CONFIG_DIR,SCHEMA_FILE)



# Config File path
ROOT_DIR=os.getcwd()
CONFIG_DIR='config'
SCHEMA_FILE='config.yaml'
CONFIG_FILE_PATH=os.path.join(ROOT_DIR,CONFIG_DIR,SCHEMA_FILE)


# Data Ingestion 
# Data Ingestion related variable
DATA_INGESTION_CONFIG_KEY = "data_ingestion_config"
DATA_INGESTION_DATABASE_NAME= "data_base"
DATA_INGESTION_COLLECTION_NAME= "collection_name"
DATA_INGESTION_ARTIFACT_DIR = "data_ingestion"
DATA_INGESTION_RAW_DATA_DIR_KEY = "raw_data_dir"
DATA_INGESTION_INGESTED_DIR_NAME_KEY = "ingested_dir"
DATA_INGESTION_TRAIN_DIR_KEY = "ingested_train_dir"
DATA_INGESTION_TEST_DIR_KEY = "ingested_test_dir"
CONFIG_FILE_KEY = "config"



# Data Validation related variable
DATA_VALIDATION_ARTIFACT_DIR="data_validation_dir"
DATA_VALIDATION_CONFIG_KEY = "data_validation_config"
DATA_VALIDATION_SCHEMA_FILE_NAME_KEY = "schema_file_name"
DATA_VALIDATION_SCHEMA_DIR_KEY = "schema_dir"
DATA_VALIDATION_VALID_DATASET ="validated_data"
DATA_VALIDATION_TRAIN_FILE = "Train_data"
DATA_VALIDATION_TEST_FILE ="Test_data"



# key  ---> config.yaml---->values
DATA_TRANSFORMATION_CONFIG_KEY = "data_transformation_config"
DATA_TRANSFORMATION='data_transformation_dir'
DATA_TRANSFORMATION_DIR_NAME_KEY = "transformed_dir"
DATA_TRANSFORMATION_TRAIN_DIR_NAME_KEY = "transformed_train_dir"
DATA_TRANSFORMATION_TEST_DIR_NAME_KEY = "transformed_test_dir"
DATA_TRANSFORMATION_PREPROCESSING_DIR_KEY = "preprocessing_dir"
DATA_TRANSFORMATION_PREPROCESSING_FILE_NAME_KEY = "preprocessed_object_file_name"

PIKLE_FOLDER_NAME_KEY = "prediction_files"

# Model Training 
MODEL_TRAINING_CONFIG_KEY='model_trainer_config'
MODEL_TRAINER_ARTIFACT_DIR = "model_training"
MODEL_TRAINER_OBJECT = "model_object"
MODEL_REPORT_FILE="model_report"


# model evaluation 
MODEL_EVAL_CONFIG_KEY='model_eval_config'
MODEL_EVALUATION_DIRECTORY='model_eval_dir'
MODEL_REPORT='model_eval_report'

# Saved Model 
SAVED_MODEL_CONFIG_KEY='saved_model_config'
SAVED_MODEL_DIR='directory'
SAVED_MODEL_OBJECT='model_object'
SAVED_MODEL_REPORT='report'