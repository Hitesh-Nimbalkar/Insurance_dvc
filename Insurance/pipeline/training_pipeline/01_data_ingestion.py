
import uuid
from Insurance.config import *
from Insurance.exception import InsuranceException
from typing import List
from Insurance.utils import read_yaml_file
from multiprocessing import Process
from Insurance.entity.config_entity import *
from Insurance.entity.artifact_entity import *
from Insurance.components.data_ingestion import DataIngestion

import  sys
from collections import namedtuple




class Pipeline():

    def __init__(self,training_pipeline_config=TrainingPipelineConfig()) -> None:
        try:
            
            self.training_pipeline_config=training_pipeline_config
            
            
            
            
        except Exception as e:
            raise InsuranceException(e, sys) from e

    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            data_ingestion = DataIngestion(data_ingestion_config=DataIngestionConfig(self.training_pipeline_config))
            data_ingestion.initiate_data_ingestion()
            
        except Exception as e:
            raise InsuranceException(e, sys) from e
        
        
if __name__ == '__main__':
    pipeline=Pipeline()
    pipeline.start_data_ingestion()
        
     