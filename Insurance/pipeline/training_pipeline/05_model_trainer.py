
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
from Insurance.components.paramter_optimize import param_optimsation
from Insurance.components.model_trainer import ModelTrainer

import  sys
from collections import namedtuple




class model_trainer():

    def __init__(self,training_pipeline_config=TrainingPipelineConfig()) -> None:
        try:
            
            self.training_pipeline_config=training_pipeline_config

            artifact=read_yaml_file(ARTIFACT_ENTITY_YAML_FILE_PATH)
            data_transformation_artifact=artifact['data_transformation_artifact']
            target_test=data_transformation_artifact['target_test']
            target_train=data_transformation_artifact['target_train']
            transform_object_path=data_transformation_artifact['transform_object_path']
            transformed_test_path=data_transformation_artifact['transformed_test_path']
            transformed_train_path=data_transformation_artifact['transformed_train_path']
            
            param_file_data=artifact['param_optimisation_artifact']
            param_file_path=param_file_data['param_file_path']
            
            
            
            model_trainer = ModelTrainer(model_training_config=ModelTrainingConfig(self.training_pipeline_config),
                                    param_artifact=OptimisationArtifact(param_yaml_file_path=param_file_path),
                                        data_transformation_artifact=DataTransformationArtifact(transform_object_path=transform_object_path,
                                                                                                transformed_train_path=transformed_train_path,
                                                                                                target_test=target_test,
                                                                                                target_train=target_train,
                                                                                                transformed_test_path=transformed_test_path))   

            return model_trainer.start_model_training()
        except Exception as e:
            raise InsuranceException(e,sys) from e  
        
if __name__ == '__main__':
    model_trainer()
        