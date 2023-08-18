            
            
import yaml           
import shutil
import os           
import sys 
from Insurance.logger import logging
from Insurance.exception import InsuranceException  
from Insurance.entity.config_entity import *         
from Insurance.entity.artifact_entity import *
from Insurance.utils import load_object,save_object,add_dict_to_yaml
from Insurance.constant import *     
from Insurance.constant import *       
import mlflow
from dotenv import load_dotenv

    

 

class ModelPusher:

    def __init__(self,model_eval_artifact:ModelEvaluationArtifact):

        try:
            self.model_eval_artifact = model_eval_artifact
            
            self.saved_model_config=SavedModelConfig()
            self.saved_model_dir=self.saved_model_config.saved_model_dir
            
        except  Exception as e:
            raise InsuranceException(e, sys)
        

    def initiate_model_pusher(self):
        try:
            # Selected model path
            eval_report_path = self.model_eval_artifact.model_eval_report_path
            
            eval_data=read_yaml_file(eval_report_path)
            # Model Path 
            model_uri=eval_data['Model_uri']

            logging.info(f" Model path : {model_uri}")
            file_path=os.path.join(self.saved_model_dir,'model.pkl')
            model = mlflow.sklearn.load_model(model_uri)
            logging.info(f"Selected model path {file_path}")
            
            save_object(file_path=file_path, obj=model)
            logging.info("Model saved.")

            os.makedirs(self.saved_model_dir, exist_ok=True)
            logging.info(f"Report Location: {self.saved_model_dir}")

            # Save the report as a YAML file
            file_path = os.path.join(self.saved_model_dir, 'report.yaml')
            with open(file_path, 'w') as file:
                yaml.dump(eval_data, file)

            logging.info("Report saved as YAML file.")
            
            model_pusher_artifact={'model_pusher_artifact': "Model Pushed succeessfully" }

            add_dict_to_yaml(file_path=ARTIFACT_ENTITY_YAML_FILE_PATH,new_data=model_pusher_artifact)

        
        except  Exception as e:
            raise InsuranceException(e, sys)
    
        
    def __del__(self):
        logging.info(f"\n{'*'*20} Model Pusher log completed {'*'*20}\n\n")
            
            
            
            
            
            
            
            
            
            
 