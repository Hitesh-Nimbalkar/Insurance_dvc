
from Insurance.exception import InsuranceException
from Insurance.logger import logging
import sys
import os
from Insurance.utils import *
from Insurance.entity.config_entity import *
from Insurance.entity.artifact_entity import *
from Insurance.constant import *
import mlflow
from Insurance.utils import read_yaml_file,load_object
from Insurance.constant import *
from mlflow.tracking import MlflowClient



class Experiments_evaluation:
    def __init__(self) :
        pass
    # Function to save run details to YAML file
    def save_run_details_to_yaml(self,run_id,experiment,model_name,best_model_path):
        client = MlflowClient()
        run = client.get_run(run_id=run_id)
        
        report = {
            "Experiment": experiment,
            "Model_name":model_name,
            "run_id": run.info.run_id,
            "experiment_id": run.info.experiment_id,
            "params": run.data.params,
            "metrics": run.data.metrics,
            "Model_path": best_model_path
        }
            
        return report
            
            
    def get_best_model_run_id(self,experiment_name, metric_name):
        # Get the experiment ID
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
        
        # Retrieve runs and sort by the specified metric
        runs = mlflow.search_runs(experiment_ids=[experiment_id], filter_string='', order_by=[f"metrics.{metric_name} DESC"])
        
        if runs.empty:
            print("No runs found for the specified experiment and metric.")
            return None
        
        # Get the best run
        best_run = runs.iloc[0]
        best_run_id = best_run.run_id
        
        # Load the best model
      #  best_model = mlflow.sklearn.load_model(f"runs:/{best_run_id}/model")
        best_model_path=(f"runs:/{best_run_id}/model")
        return best_model_path,best_run_id

    def run_mlflow_experiment(self,experiment_name, run_name, R2_score, parameters, model):
        """
        Run an MLflow experiment, log metrics, parameters, and a model,
        and return the best model and its run ID based on the specified metric.
        
        Parameters:
        - experiment_name (str): Name of the MLflow experiment.
        - run_name (str): Name of the run.
        - R2_score (float): R2 score to be logged as a metric.
        - parameters (dict): Dictionary of parameters to be logged.
        - model: The machine learning model to be logged.
        
        Returns:
        - best_model (object): The best model from the experiment.
        - best_run_id (str): ID of the best run based on the specified metric.
        """
        # Create or get the experiment
        mlflow.set_experiment(experiment_name)
        
        # Start a run
        with mlflow.start_run(run_name=run_name):
            # Log metrics, params, and model
            mlflow.log_metric("R2_score", float(R2_score))
            mlflow.log_params(parameters)
            mlflow.sklearn.log_model(model, artifact_path='model')
        
        best_model_path, best_run_id = self.get_best_model_run_id(metric_name='R2_score', experiment_name=experiment_name)
        
        print(f"Best model Run id: {best_run_id}")
        
        return best_model_path, best_run_id



class ModelEvaluation:


    def __init__(self,model_evaluation_config:ModelEvalConfig,
                    model_trainer_artifact:ModelTrainerArtifact):
        
        try:

    
            self.model_trainer_artifact=model_trainer_artifact
            self.model_evaluation_config=model_evaluation_config

            self.saved_model_config=SavedModelConfig()
            
            self.saved_model_directory=self.saved_model_config.saved_model_dir
            
        except Exception as e:
            raise InsuranceException(e,sys)
        
        
        
    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            logging.info(" Model Evaluation Started ")
            ## Artifact trained Model  files
            model_trained_artifact_path = self.model_trainer_artifact.model_object_file_path
            model_trained_report = self.model_trainer_artifact.model_report_file_path
            
            
            # Model evaludation report Directoy 
            os.makedirs(self.model_evaluation_config.model_eval_directory,exist_ok=True)
               
            logging.info(f" Artifact Trained model :")
            
            params_yaml_data=read_yaml_file('params.yaml')
            experiment_name=params_yaml_data['Experiment']
            run_name=params_yaml_data['run_name']
            
            Exp_eval=Experiments_evaluation()

            # Loading the models
            logging.info("Saved_models directory .....")
            os.makedirs(self.saved_model_directory,exist_ok=True)
            

                
            # Report 
            model_trained_report_data = read_yaml_file(file_path=model_trained_report)
            
            artifact_model_r2_score =float( model_trained_report_data['R2_score'])
            artifact_model_params=model_trained_report_data['Parameters']
            model_name = model_trained_report_data['Model_name']
            R2_score = artifact_model_r2_score
            
            best_model_path, best_run_id=Exp_eval.run_mlflow_experiment(experiment_name=experiment_name,
                                                    run_name=run_name,
                                                    R2_score=R2_score,
                                                    parameters=artifact_model_params,
                                                    model=model_trained_artifact_path)
            
            report=Exp_eval.save_run_details_to_yaml(
                                              run_id=best_run_id,
                                              experiment=experiment_name,model_name=model_name,best_model_path=best_model_path)
        
            
            write_yaml_file(file_path=self.model_evaluation_config.model_eval_report,data=report)
            
            artifact_report_path=self.model_evaluation_config.model_eval_report
            model_evaluation_artifact= {'model_evaluation_artifact': {'eval_report':artifact_report_path}}
            
            add_dict_to_yaml(file_path=ARTIFACT_ENTITY_YAML_FILE_PATH,new_data=model_evaluation_artifact)

        
        except Exception as e:
            logging.error("Error occurred during model evaluation!")
            raise InsuranceException(e, sys) from e


    def __del__(self):
        logging.info(f"\n{'*'*20} Model evaluation log completed {'*'*20}\n\n")
