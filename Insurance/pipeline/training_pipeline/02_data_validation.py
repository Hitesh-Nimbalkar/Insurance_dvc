import uuid
from Insurance.config import *
from Insurance.exception import InsuranceException
from typing import List
from Insurance.utils import read_yaml_file
from multiprocessing import Process
from Insurance.entity.config_entity import *
from Insurance.entity.artifact_entity import DataValidationArtifact,DataIngestionArtifact
from Insurance.components.data_ingestion import DataIngestion
from Insurance.components.data_validation import DataValidation
from Insurance.components.data_transformation import DataTransformation

from Insurance.components.model_trainer import ModelTrainer

from Insurance.components.model_evaluation import ModelEvaluation
from Insurance.components.model_pusher import ModelPusher

import  sys
from collections import namedtuple




class data_validation():

    def __init__(self,training_pipeline_config=TrainingPipelineConfig()) -> None:
        try:
            self.training_pipeline_config=training_pipeline_config
                       
            artifact=read_yaml_file(ARTIFACT_ENTITY_YAML_FILE_PATH)
            data_ingestion_artifact=artifact['data_ingestion_artifact']
            train_path=data_ingestion_artifact['train_file_path']
            test_path=data_ingestion_artifact['test_file_path']
            
            data_validation = DataValidation(data_validation_config=DataValidationConfig(self.training_pipeline_config),
                                             data_ingestion_artifact=DataIngestionArtifact(train_file_path=train_path,
                                                                                           test_file_path=test_path))
            data_validation.initiate_data_validation()
        except Exception as e:
            raise InsuranceException(e, sys) from e
        
     

if __name__ == '__main__':
    data_validation()
        