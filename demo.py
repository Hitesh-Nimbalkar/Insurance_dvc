
from Insurance.config import *

from Insurance.logger import logging
from Insurance.pipeline.train import Pipeline


def main():
    try:
        pipeline = Pipeline()
        pipeline.run_pipeline()

    except Exception as e:
        logging.error(f"{e}")
        print(e)

if __name__ == "__main__":
    main()
