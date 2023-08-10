
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

    def start_model_evaluation(self):
        try:
            artifact=read_yaml_file(ARTIFACT_ENTITY_YAML_FILE_PATH)
            model_object=artifact['model_file_path']
            report_path=artifact['Report_path']
            
            
            model_eval = ModelEvaluation(
                model_trainer_artifact=ModelTrainerArtifact(model_object_file_path=model_object,model_report_file_path=report_path),
                model_evaluation_config=ModelEvalConfig(training_pipeline_config=self.training_pipeline_config))
                                         
            model_eval.initiate_model_evaluation()
         
        except  Exception as e:
            raise  InsuranceException(e,sys)
        
        
pipeline=Pipeline()
pipeline.start_model_evaluation()