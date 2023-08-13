            
            
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

class MLFlowExperiment:
    def __init__(self, experiment_name):
        self.experiment_name = experiment_name
        self.experiment_id = None

    def create_experiment(self):
        self.experiment_id = mlflow.create_experiment(self.experiment_name)
        print(f"Experiment '{self.experiment_name}' created with ID: {self.experiment_id}")

    def log_metrics(self, metrics):
        if self.experiment_id is None:
            print("Experiment not created. Call 'create_experiment()' first.")
            return

        with mlflow.start_run(experiment_id=self.experiment_id):
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
                print(f"Logged metric: {metric_name} = {metric_value}")

    def log_parameters(self, parameters):
        if self.experiment_id is None:
            print("Experiment not created. Call 'create_experiment()' first.")
            return

        with mlflow.start_run(experiment_id=self.experiment_id):
            for param_name, param_value in parameters.items():
                mlflow.log_param(param_name, param_value)
                print(f"Logged parameter: {param_name} = {param_value}")

    def log_model(self, model_path, model_name):
        if self.experiment_id is None:
            print("Experiment not created. Call 'create_experiment()' first.")
            return

        with mlflow.start_run(experiment_id=self.experiment_id):
            mlflow.log_artifact(model_path, artifact_path="models")
            print(f"Logged model: {model_name}")


            

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
            
            # Model Report
            model_name=eval_data['Model_name']
            model_path=eval_data['Model_path']
            R2_score=eval_data['R2_score']
            report_path=eval_data['Report_path']
            parameters=eval_data['Parameters']
            
            logging.info(f" Model path : {model_path}")
            file_path=os.path.join(self.saved_model_dir,'model.pkl')
            model = load_object(file_path=model_path)
            logging.info(f"Selected model path {file_path}")
            
            save_object(file_path=file_path, obj=model)
            logging.info("Model saved.")
            
            params_yaml_data=read_yaml_file('params.yaml')
            experiment_name=params_yaml_data['Experiment']

            mlflow_experiment = MLFlowExperiment(experiment_name)
            mlflow_experiment.create_experiment()
            
            mlflow_experiment.log_metrics({'R2_score': R2_score})
            mlflow_experiment.log_parameters(parameters)
            mlflow_experiment.log_model(model_path, model_name)

                
            
            # Create a dictionary for the report
            report = {'Model_name': model_name, 'R2_score': R2_score,
                      'Parameters': parameters,'Report_path': report_path}

            logging.info(str(report))
            
            os.makedirs(self.saved_model_dir, exist_ok=True)
            logging.info(f"Report Location: {self.saved_model_dir}")

            # Save the report as a YAML file
            file_path = os.path.join(self.saved_model_dir, 'report.yaml')
            with open(file_path, 'w') as file:
                yaml.dump(report, file)

            logging.info("Report saved as YAML file.")
            
            model_pusher_artifact={'model_pusher_artifact': "Model Pushed succeessfully" }

            add_dict_to_yaml(file_path=ARTIFACT_ENTITY_YAML_FILE_PATH,new_data=model_pusher_artifact)

        
        except  Exception as e:
            raise InsuranceException(e, sys)
    
        
    def __del__(self):
        logging.info(f"\n{'*'*20} Model Pusher log completed {'*'*20}\n\n")
            
            
            
            
            
            
            
            
            
            
 