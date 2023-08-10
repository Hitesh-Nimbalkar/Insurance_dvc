            
            
import yaml           
import shutil
import os           
import sys 
from Insurance.logger import logging
from Insurance.exception import InsuranceException  
from Insurance.entity.config_entity import *         
from Insurance.entity.artifact_entity import *
from Insurance.utils import load_object,save_object
from Insurance.constant import *     
from Insurance.constant import *       

            
            
            
            
          
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
            model_path = self.model_eval_artifact.selected_model_path
            logging.info(f" Model path : {model_path}")
            file_path=os.path.join(self.saved_model_dir,'model.pkl')
            model = load_object(file_path=model_path)
            logging.info(f"Selected model path {file_path}")
            
            save_object(file_path=file_path, obj=model)
            logging.info("Model saved.")
            
            # Model report
            model_name = self.model_eval_artifact.model_name
            R2_score = self.model_eval_artifact.R2_score
           
            
            
            # Create a dictionary for the report
            report = {'Model_name': model_name, 'R2_score': R2_score}

            logging.info(str(report))
            
            os.makedirs(self.saved_model_dir, exist_ok=True)
            logging.info(f"Report Location: {self.saved_model_dir}")

            # Save the report as a YAML file
            file_path = os.path.join(self.saved_model_dir, 'report.yaml')
            with open(file_path, 'w') as file:
                yaml.dump(report, file)

            logging.info("Report saved as YAML file.")
            
            
        

            model_pusher_artifact = ModelPusherArtifact(message="Model Pushed succeessfully")
            return model_pusher_artifact
        except  Exception as e:
            raise InsuranceException(e, sys)
    
            
            
    def __del__(self):
        logging.info(f"\n{'*'*20} Model Pusher log completed {'*'*20}\n\n")
            
            
            
            
            
            
            
            
            
            
 