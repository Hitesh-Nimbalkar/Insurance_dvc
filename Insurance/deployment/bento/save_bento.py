
import bentoml
from Insurance.utils import load_object
from Insurance.logger import logging


class create_bento:
    
    def __init__(self,model_path) -> None:
        self.model_path=model_path
        
    def load_model_and_save_it_to_bento(self):
        """Loading Models and save as bento """
        model = load_object(self.model_path)
        bento_model = bentoml.sklearn.save_model("sklearn_model", model)
        logging.info(f"Bento model tag = {bento_model.tag}")
        
        return bento_model.tag



