
from pathlib import Path
from Insurance.utils import load_object
from Insurance.logger import logging
import bentoml


def load_model_and_save_it_to_bento(model_path: Path) -> None:
    """Loads a keras model from disk and saves it to BentoML."""
    model = load_object(model_path)
    bento_model = bentoml.sklearn.save_model("sklearn_model", model)
    logging.info(f"Bento model tag = {bento_model.tag}")


if __name__ == "__main__":
    load_model_and_save_it_to_bento(Path("model"))