import pymongo # pip install pymongo
import pandas as pd
import json
import urllib
import os
import yaml
from Insurance.data_base import mongo_client


client = mongo_client()

DATA_FILE_PATH = (r"E:\Projects\Insurance_DB\insurance.csv")
DATABASE_NAME = "INSURANCE"
COLLECTION_NAME = "INSURANCE_PROJECT"


if __name__=="__main__":
    df = pd.read_csv(DATA_FILE_PATH)
    print(f"Rows and columns: {df.shape}")

    df.reset_index(drop = True, inplace = True)

    json_record = list(json.loads(df.T.to_json()).values())
    print(json_record[0])

    client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_record)