
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
    def __init__(self,experiment_name,run_name) :
        self.experiment_name=experiment_name
        self.run_name=run_name
    # Function to save run details to YAML file
    def save_run_details_to_yaml(self,run_id,model_name,model_uri,run_name):
        client = MlflowClient()
        run = client.get_run(run_id=run_id)
        
        report = {
            "Experiment": self.experiment_name,
            "Model_name":model_name,
            "run_name":run_name,
            "run_id": run.info.run_id,
            "experiment_id": run.info.experiment_id,
            "Parameters": run.data.params,
            "metrics": run.data.metrics,
            "Model_uri": model_uri
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
        model_uri=(f"runs:/{best_run_id}/model")
        return model_uri,best_run_id

    def run_mlflow_experiment(self, R2_score, parameters, model_path):
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
        mlflow.set_experiment(self.experiment_name)
        
        # Start a run
        with mlflow.start_run(run_name=self.run_name):
            # Log metrics, params, and model
            mlflow.log_metric("R2_score", float(R2_score))
            mlflow.log_params(parameters)
             # Load your trained model
            model = load_object(model_path)
            mlflow.sklearn.log_model(model, "model")
        
        model_uri,best_run_id = self.get_best_model_run_id(metric_name='R2_score', experiment_name=self.experiment_name)
        
        print(f"Best model Run id: {best_run_id}")
        
        return   model_uri,best_run_id



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
            
            Exp_eval=Experiments_evaluation(experiment_name=experiment_name,run_name=run_name)

            # Saved Model files
            saved_model_path = self.saved_model_config.saved_model_file_path
            saved_model_report_path=self.saved_model_config.saved_model_report_path

            # Loading the models
            logging.info("Saved_models directory .....")
            os.makedirs(self.saved_model_directory,exist_ok=True)
            
            # Check if SAVED_MODEL_DIRECTORY is empty
            if not os.listdir(self.saved_model_directory):
                # Report 
                model_trained_report_data = read_yaml_file(file_path=model_trained_report)
                
                R2_score =float( model_trained_report_data['R2_score'])
                artifact_model_params=model_trained_report_data['Parameters']
                model_name = model_trained_report_data['Model_name']
                
                model_uri,best_run_id=Exp_eval.run_mlflow_experiment(R2_score=R2_score,
                                                                                parameters=artifact_model_params,
                                                                                model_path=model_trained_artifact_path)
                                        
                report=Exp_eval.save_run_details_to_yaml(run_id=best_run_id,
                                                         run_name=run_name,
                                                        model_name=model_name,
                                                        model_uri=model_uri
                                                        )             
            else:
                saved_model_report_data = read_yaml_file(file_path=saved_model_report_path)
                model_trained_report_data = read_yaml_file(file_path=model_trained_report)
                
                # Compare the R2_scores and accuracy of the two models
                saved_model_r2_score = float(saved_model_report_data['metrics']['R2_score'])
                artifact_model_R2_score =float(model_trained_report_data['R2_score'])
                       
                # Compare the models and log the result
                if artifact_model_R2_score > saved_model_r2_score:
                    logging.info("Trained model outperforms the saved model!")
                    R2_score =float( model_trained_report_data['metrics']['R2_score'])
                    artifact_model_params=model_trained_report_data['Parameters']
                    model_name = model_trained_report_data['Model_name']
                    
                    model_uri,best_run_id=Exp_eval.run_mlflow_experiment(R2_score=R2_score,
                                                                                    parameters=artifact_model_params,
                                                                                    model=model_trained_artifact_path)
                    
                    comment='Trained Model is better than Saved model'
                    
                    report=Exp_eval.save_run_details_to_yaml(run_id=best_run_id,
                                                             model_name=model_name,
                                                            model_uri=model_uri
                                                            )
                    
                    
                    
                    logging.info(f"Model Selected : {model_name}")
                    logging.info(f"R2_score : {R2_score}")
                    
                elif saved_model_r2_score>artifact_model_R2_score:
                    logging.info("Trained model underperforms the saved model!")
                    R2_score =float( saved_model_report_data['metrics']['R2_score'])
                    saved_model_params=saved_model_report_data['Parameters']
                    model_name = saved_model_report_data['Model_name']
                    experiment_name=saved_model_report_data['Experiment']
                    run_name=saved_model_report_data['run_name']
           
                    
                    Exp_eval=Experiments_evaluation(experiment_name=experiment_name,run_name=run_name)
                    model_uri,best_run_id=Exp_eval.run_mlflow_experiment(R2_score=R2_score,
                                                                                    parameters=saved_model_params,
                                                                                    model=saved_model_path)
                    
                    
                    comment='Saved Model is better than Trained model'
                             
                    report=Exp_eval.save_run_details_to_yaml(run_id=best_run_id,
                                                             model_name=model_name,
                                                            model_uri=model_uri
                                                            )
                    
                    
                    
                    logging.info(f"Model Selected : {model_name}")
                    logging.info(f"R2_score : {R2_score}")
                
                else:
                    logging.info("Trained model underperforms the saved model!")
                    R2_score =float( saved_model_report_data['metrics']['R2_score'])
                    saved_model_params=saved_model_report_data['Parameters']
                    model_name = saved_model_report_data['Model_name']
                    experiment_name=saved_model_report_data['Experiment']
                    run_name=saved_model_report_data['run_name']

                    
                    Exp_eval=Experiments_evaluation(experiment_name=experiment_name,run_name=run_name)
                    model_uri,best_run_id=Exp_eval.run_mlflow_experiment(R2_score=R2_score,
                                                                                    parameters=saved_model_params,
                                                                                    model_path=saved_model_path)
                                            
                    report=Exp_eval.save_run_details_to_yaml(run_name=run_name,
                                                            run_id=best_run_id,
                                                             model_name=model_name,
                                                            model_uri=model_uri
                                                            )
                    
                    comment='Saved Model and Trained model performs equally'
                    
                    logging.info(f"Model Selected : {model_name}")
                    logging.info(f"R2_score : {R2_score}")
                    
                    
                
            write_yaml_file(file_path=self.model_evaluation_config.model_eval_report,data=report)
            
            artifact_report_path=self.model_evaluation_config.model_eval_report
            model_evaluation_artifact= {'model_evaluation_artifact': {'eval_report':artifact_report_path}}
            
            add_dict_to_yaml(file_path=ARTIFACT_ENTITY_YAML_FILE_PATH,new_data=model_evaluation_artifact)

        
        except Exception as e:
            logging.error("Error occurred during model evaluation!")
            raise InsuranceException(e, sys) from e


    def __del__(self):
        logging.info(f"\n{'*'*20} Model evaluation log completed {'*'*20}\n\n")
