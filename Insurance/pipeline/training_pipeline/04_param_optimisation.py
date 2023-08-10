
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

    def param_optimisation(self) -> ModelTrainerArtifact:
        try:
            artifact=read_yaml_file(ARTIFACT_ENTITY_YAML_FILE_PATH)
            data_transformation_artifact=artifact['data_transformation_artifact']
            target_test=data_transformation_artifact['target_test']
            target_train=data_transformation_artifact['target_train']
            transform_object_path=data_transformation_artifact['transform_object_path']
            transformed_test_path=data_transformation_artifact['transformed_test_path']
            transformed_train_path=data_transformation_artifact['transformed_train_path']
            
            
            
            optimise = param_optimsation(paramter_optimise_config=Hyperparamter_optimize(self.training_pipeline_config),
                                        data_transformation_artifact=DataTransformationArtifact(transform_object_path=transform_object_path,
                                                                                                transformed_train_path=transformed_train_path,
                                                                                                target_test=target_test,
                                                                                                target_train=target_train,
                                                                                                transformed_test_path=transformed_test_path))   

            return optimise.start_param_optimisation()
        except Exception as e:
            raise InsuranceException(e,sys) from e  
        
        
pipeline=Pipeline()
pipeline.param_optimisation()
        
        