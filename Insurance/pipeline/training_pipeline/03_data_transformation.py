
import uuid
from Insurance.config import *
from Insurance.exception import InsuranceException
from typing import List
from Insurance.utils import read_yaml_file
from multiprocessing import Process
from Insurance.entity.config_entity import *
from Insurance.entity.artifact_entity import *
from Insurance.components.data_ingestion import DataIngestion
from Insurance.components.data_validation import DataValidation
from Insurance.components.data_transformation import DataTransformation

from Insurance.components.model_trainer import ModelTrainer

from Insurance.components.model_evaluation import ModelEvaluation
from Insurance.components.model_pusher import ModelPusher

import  sys
from collections import namedtuple




class Pipeline():

    def __init__(self,training_pipeline_config=TrainingPipelineConfig()) -> None:
        try:
            
            self.training_pipeline_config=training_pipeline_config

        except Exception as e:
            raise InsuranceException(e, sys) from e
        
       
    def start_data_transformation(self) -> DataTransformationArtifact:
        try:
            artifact=read_yaml_file(ARTIFACT_ENTITY_YAML_FILE_PATH)
            data_validation_artifact=artifact['data_validation_artifact']
            train_path=data_validation_artifact['validated_train_path']
            test_path=data_validation_artifact['validated_test_path']
            
            
            data_transformation = DataTransformation(
                data_transformation_config = DataTransformationConfig(self.training_pipeline_config),
                data_validation_artifact = DataValidationArtifact(validated_test_path=test_path,
                                                                  validated_train_path=train_path))

            return data_transformation.initiate_data_transformation()
        except Exception as e:
            raise InsuranceException(e,sys) from e
        
if __name__ == '__main__':
    pipeline=Pipeline()
    pipeline.start_data_transformation()
        