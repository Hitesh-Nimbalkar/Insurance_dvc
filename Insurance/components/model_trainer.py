
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
    def __init__(self,X_train,X_test,y_train,y_test):
        self.models_dict = {
            "Random Forest Regression": RandomForestRegressor(),
            "XGBoost Regression": xgb.XGBRegressor(),
            "Gradient Boosting Regression": GradientBoostingRegressor()
        }
        # Parameter grid for each model
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
        
        self.X_train=X_train
        self.X_test=X_test
        self.y_train=y_train
        self.y_test=y_test
        
        self.best_model_dictionary = {}
        self.model_and_params = {}

    def train_models(self):
        # Train each model and evaluate its performance
        results = {}
        best_model_label = None
        best_r2_score = -float('inf')

        for model_label, model in self.models_dict.items():
            
            
            model.fit(self.X_train, self.y_train)  # Train the model
            y_pred = model.predict(self.X_test)  # Make predictions
            r2 = r2_score(self.y_test, y_pred)  # Calculate R2 score
            results[model_label] = r2  # Store the R2 score in the results dictionary

            logging.info(f" R2 Score  : {model_label} -{r2}")
            
            
            # Keep track of the best model
            if r2 > best_r2_score:
                best_r2_score = r2
                best_model_label = model_label
                
        logging.info(f"Model Selected - {model_label} R2 Score : {r2}")

        # Create the best model dictionary
        self.best_model_dictionary = {best_model_label: self.models_dict[best_model_label]}
    
    def train_model_selected(self):
        # Train model with the best hyperparameters and store the fitted models, best parameters, and model labels

        for model_label, model in self.best_model_dictionary.items():
            # Hyperparameter tuning with GridSearchCV
            grid_search = GridSearchCV(model, self.param_grid[model_label], cv=5)
            grid_search.fit(self.X_train, self.y_train)
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_

            # Fit the best model on the whole training data
            best_model.fit(self.X_train, self.y_train)

            # Save the fitted model, best parameters, and model label
            self.model_and_params= {
                'model_label': model_label,
                'model': best_model,
                'best_params': best_params
                
            }
            
            
            
            logging.info(f" { self.model_and_params}")
            
    def predict_with_model(self):
    
        
        self.model_and_params
        
        logging.info(f" Prediction using Model :  {self.model_and_params['model_label']} ")
        
        logging.info(f" Best Params : {self.model_and_params['best_params']} ")

        model=self.model_and_params['model']
        
        # Make predictions using the specified model
        predictions = model.predict(self.X_test)
        
        logging.info(" X_test Prediction complete ")
        
        R2_score= r2_score(self.y_test, predictions)
        # Convert R2_score to a string with 2 decimal places
        r2_score_str = "{:.2f}".format(R2_score)

        # Create the model_report dictionary
        model_report = {
            'Model_name': self.model_and_params['model_label'],
            'R2_score': r2_score_str
        }

        return predictions,model,model_report





class ModelTrainer :


    def __init__(self,model_training_config:config_entity.ModelTrainingConfig,
                    data_transformation_artifact:artifact_entity.DataTransformationArtifact):
        try:
            self.model_training_config=model_training_config
            self.data_transformation_artifact=data_transformation_artifact
            
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
            
            # Target Variables
            
            # Model training 
            training=Regressor(X_train=X_train,X_test=X_test,y_train=y_train,y_test=y_test)
            training.train_models()
            training.train_model_selected()
            predictions,model,model_report=training.predict_with_model()
            
            
          #  logging.info(f" Predictions : {predictions}")
            
             # Save object 
            utils.save_object(file_path=self.model_training_config.model_object,obj=model)
            
            # Saving report 
            utils.write_yaml_file(file_path=self.model_training_config.model_report,data=model_report)
            
            model_trainer_artifact=artifact_entity.ModelTrainerArtifact(model_object_file_path=self.model_training_config.model_object,
                                                                model_report_file_path=self.model_training_config.model_report)
            
            
            return model_trainer_artifact
            
        except Exception as e:
            raise InsuranceException(e, sys)
        
            
        
        
        
        
        
        