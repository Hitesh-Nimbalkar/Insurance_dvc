from Insurance.utils import load_object
import bentoml
from bentoml.io import PandasDataFrame,NumpyNdarray,JSON
import numpy as np
import pandas as pd
import os
from pydantic import BaseModel

SAVED_MODEL_DIRECTORY=os.path.join(os.getcwd(),'Saved_model','model.pkl')
class input(BaseModel):
    age: float=  20.0
    sex: str = "female"
    bmi:float = 20.0 
    children: float = 20
    smoker:str= "no"
    region:str = "southeast"


# Add model to the runner 
model_runner = bentoml.sklearn.get("sklearn_model:latest").to_runner() 

print(" Model Runner SET")

# Create Service
svc = bentoml.Service("expense_predictor", runners=[model_runner])


@svc.api(input=JSON(pydantic_model=input), output=NumpyNdarray())
def expense(input_data:input) -> np.ndarray:
    df=pd.DataFrame([input_data.dict()],index=[0])  
    print("--------------------------")
    print(df)
    preprocessor=load_object(file_path=r"E:\Projects\Deployment\Notebook\BentoML\Preprocessed_files\preprocessor.pkl")
    array=preprocessor.transform(df)
    print("--------------------------")
    print(array)
    return model_runner.run(array)




