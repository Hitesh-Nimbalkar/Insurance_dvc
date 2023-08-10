
from Insurance.entity import artifact_entity,config_entity
from Insurance.exception import InsuranceException
from Insurance.logger import logging
import sys 
from Insurance import utils
from Insurance.constant import *
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

class Regressor:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.models_dict = {
            "Random Forest Regression": RandomForestRegressor(),
            "XGBoost Regression": xgb.XGBRegressor(),
            "Gradient Boosting Regression": GradientBoostingRegressor()
        }
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.fitted_model = None

    def fit_model(self, model_name, model_parameters):
        # Get the model instance based on the provided model name
        model_instance = self.models_dict.get(model_name)
        
        if model_instance is None:
            raise ValueError(f"Model '{model_name}' not found.")
        
        # Set model parameters if provided
        if model_parameters:
            model_instance.set_params(**model_parameters)
        
        # Fit the model with the training data
        model_instance.fit(self.X_train, self.y_train)
        
        self.fitted_model = model_instance
        return model_instance




class ModelTrainer :


    def __init__(self,model_training_config:config_entity.ModelTrainingConfig,
                    param_artifact:artifact_entity.OptimisationArtifact,
                    data_transformation_artifact:artifact_entity.DataTransformationArtifact):
        try:
            self.model_training_config=model_training_config
            self.data_transformation_artifact=data_transformation_artifact
            self.param_artifact=param_artifact
            
            # Accessing file location 
            self.trained_model_path=self.model_training_config.model_object
            self.trained_model_report=self.model_training_config.model_report
            
        except Exception as e:
            raise InsuranceException(e, sys)
        
    def start_model_training(self):
        
        try:

            
            y_train=utils.load_numpy_array_data(file_path=self.data_transformation_artifact.target_train)
            y_test=utils.load_numpy_array_data(file_path=self.data_transformation_artifact.target_test)
            
            X_train=utils.load_numpy_array_data(self.data_transformation_artifact.transformed_train_path)
            X_test=utils.load_numpy_array_data(self.data_transformation_artifact.transformed_test_path)
            
            # Params yaml file 
            params_data = utils.read_yaml_file(self.param_artifact.param_yaml_file_path)
            
            model_name=params_data['best_model']
            parameters=params_data['params']
            R2_score=params_data['R2_score']
            
            
            # Model training 
            training=Regressor(X_train=X_train,X_test=X_test,y_train=y_train,y_test=y_test)
            model=training.fit_model(model_name=model_name,model_parameters=parameters)
            
            
            model_report ={
                "Model_name": model_name,
                "Parameters" : parameters,
                "R2_score"  : R2_score
                
                
    
            }
            
            model_trainer_artifact={'model_trainer_artifact': { 'model_file_path' :self.model_training_config.model_object,
                                                               'Report_path' :self.model_training_config.model_report
                
            }}
            
            logging.info(" Model  Fitted  ")
                


            utils.add_dict_to_yaml(file_path=ARTIFACT_ENTITY_YAML_FILE_PATH,new_data=model_trainer_artifact)

 
        
            model_trainer_directory=self.model_training_config.model_training_dir
            os.makedirs(model_trainer_directory,exist_ok=True)
            
             # Save object 
            utils.save_object(file_path=self.model_training_config.model_object,obj=model)
            
            # Saving report 
            utils.write_yaml_file(file_path=self.model_training_config.model_report,data=model_report)

        
            
        except Exception as e:
            raise InsuranceException(e, sys)
        
            
        
        
        
        
        
        