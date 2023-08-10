from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    train_file_path:str 
    test_file_path:str

@dataclass
class DataValidationArtifact:
    validated_train_path:str
    validated_test_path:str

@dataclass
class DataTransformationArtifact:
    transform_object_path:str
    transformed_train_path:str
    transformed_test_path:str
    target_train:str
    target_test:str
    
@dataclass
class OptimisationArtifact:
    param_yaml_file_path:str
    
@dataclass
class ModelTrainerArtifact:
    model_object_file_path:str
    model_report_file_path:str

@dataclass
class ModelEvaluationArtifact:
    model_name:str
    R2_score:float
    selected_model_path:str
    model_report_path:str


@dataclass
class ModelPusherArtifact:
    message:str
