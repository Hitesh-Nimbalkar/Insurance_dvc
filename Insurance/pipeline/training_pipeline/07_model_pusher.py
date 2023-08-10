
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

    def start_model_pusher(self):
            try:
                artifact=read_yaml_file(ARTIFACT_ENTITY_YAML_FILE_PATH)
                model_evaluation_artifact=artifact['model_evaluation_artifact']
                model_eval_path=model_evaluation_artifact['eval_report']

                
                
                
                
                
                model_pusher = ModelPusher(model_eval_artifact=ModelEvaluationArtifact(model_eval_report_path=model_eval_path
                                                                                       ))
                model_pusher_artifact = model_pusher.initiate_model_pusher()
                return model_pusher_artifact
            except  Exception as e:
                raise  InsuranceException(e,sys)
            
pipeline=Pipeline()
pipeline.start_model_pusher()