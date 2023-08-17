

from Insurance.entity import artifact_entity,config_entity
from Insurance.exception import InsuranceException
from Insurance.logger import logging
import sys 
import pandas as pd
from Insurance import utils
from Insurance.constant import *
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
import yaml

class Regressor:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        self.models_dict = {
            "Random Forest Regression": RandomForestRegressor(),
            "XGBoost Regression": xgb.XGBRegressor(),
            "Gradient Boosting Regression": GradientBoostingRegressor()
        }

        self.param_grid = {
            "Random Forest Regression": {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5, 10]
            },
            "XGBoost Regression": {
                "learning_rate": [0.1, 0.01],
                "max_depth": [3, 5, 7],
                "n_estimators": [50, 100, 200]
            },
            "Gradient Boosting Regression": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.1, 0.01],
                "max_depth": [3, 5, 7]
            }
        }

    def perform_hyperparameter_search(self, model_name, search_type='grid'):
        if model_name not in self.models_dict:
            return "Model not found."

        model = self.models_dict[model_name]
        param_grid = self.param_grid[model_name]

        if search_type == 'grid':
            search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)
        elif search_type == 'random':
            search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=10, cv=5, n_jobs=-1)

        search.fit(self.X_train, self.y_train)
        
        best_params = search.best_params_
        best_score = search.best_score_

        return best_params, best_score

    def perform_hyperparameter_search_all_models(self, search_type='grid'):
        results = []

        for model_name in self.models_dict:
            model = self.models_dict[model_name]
            param_grid = self.param_grid[model_name]

            if search_type == 'grid':
                search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)
            elif search_type == 'random':
                search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=10, cv=5, n_jobs=-1)

            search.fit(self.X_train, self.y_train)

            best_params = search.best_params_
            best_score = search.best_score_

            results.append({'Model': model_name, 'Best Parameters': best_params, 'R2 Score': best_score})

        df = pd.DataFrame(results)
        return df
    def choose_best_model(self, results_df):
        best_model_row = results_df.loc[results_df['R2 Score'].idxmax()]
        return best_model_row['Model'], best_model_row['Best Parameters'], best_model_row['R2 Score']




class param_optimsation:


    def __init__(self,paramter_optimise_config:config_entity.Hyperparamter_optimize,
                    data_transformation_artifact:artifact_entity.DataTransformationArtifact):
        try:
            self.paramter_optimise_config=paramter_optimise_config
            self.data_transformation_artifact=data_transformation_artifact
            
        except Exception as e:
            raise InsuranceException(e, sys)
        
    def start_param_optimisation(self):
        
        try:

            
            y_train=utils.load_numpy_array_data(file_path=self.data_transformation_artifact.target_train)
            y_test=utils.load_numpy_array_data(file_path=self.data_transformation_artifact.target_test)
            
            X_train=utils.load_numpy_array_data(self.data_transformation_artifact.transformed_train_path)
            X_test=utils.load_numpy_array_data(self.data_transformation_artifact.transformed_test_path)
            
            # Target Variables
            
            # Model training 
            training=Regressor(X_train=X_train,X_test=X_test,y_train=y_train,y_test=y_test)
            results_df=training.perform_hyperparameter_search_all_models()
            
            
            logging.info(f" Optuna HPO For models : {results_df}")
            
            
            best_model,best_model_params,R2_score=training.choose_best_model(results_df=results_df)
            
            
            logging.info(f" Best model : {best_model}")
            
            logging.info(f" Best Paramaters : {best_model_params}")
            
            logging.info(f"R2 Score : {R2_score}")
            
            if not os.path.exists('params.yaml'):
                experiment_name='Experiment_00'
                run_name='run_name'
                
            else:
                params=utils.read_yaml_file('params.yaml')
                experiment_name=params['Experiment']
                run_name=params['run_name']
                
                
            
            # Create a dictionary to hold the information
            info_dict = {
                "Experiment" : experiment_name,
                "run_name"   : run_name,
                "Model_name": best_model,
                "Parameters": best_model_params,

            }
            
            param_dir=self.paramter_optimise_config.parameter_directory
            os.makedirs(param_dir,exist_ok=True)
            
            logging.info(" Param Directory created ")
            
            param_file_path=self.paramter_optimise_config.param_file_path

            # Write the dictionary to the YAML file
            with open(param_file_path, 'w') as yaml_file:
                yaml.dump(info_dict, yaml_file, default_flow_style=False)
                
                
                
                # Write the dictionary to the YAML file
            with open('params.yaml', 'w') as yaml_file:
                yaml.dump(info_dict, yaml_file, default_flow_style=False)
                
            logging.info(" Param Dictionary Dumped ")
                
            param_optimisation_artifact = {
                'param_optimisation_artifact': {
                    'param_file_path': param_file_path
                }
            }

            utils.add_dict_to_yaml(file_path=ARTIFACT_ENTITY_YAML_FILE_PATH,new_data=param_optimisation_artifact)

            
                
        except Exception as e:
            raise InsuranceException(e, sys)
            
            
            