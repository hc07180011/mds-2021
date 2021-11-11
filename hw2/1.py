import os
import ssl
import shutil
import pickle
import zipfile
import tempfile
import urllib.request
import numpy as np
import pandas as pd


data_dir = "data"
cache_dir = ".cache"


def data_loader(
    path: str = "MiningProcess_Flotation_Plant_Database.csv"
) -> pd.DataFrame:
    os.makedirs(cache_dir, exist_ok=True)

    if os.path.exists(os.path.join(cache_dir, "{}.pickle".format(path))):
        with open(os.path.join(cache_dir, "{}.pickle".format(path)), "rb") as f:
            df = pickle.load(f)

    else:
        df = pd.read_csv(os.path.join(data_dir, path), delimiter=",")
        datetime_column = "date"
        percent_columns = list([col for col in df.columns if "%" in col])
        normal_columns = list([col for col in df.columns if "date" != col and "%" not in col])

        df = df.replace(",", ".", regex=True)
        df[datetime_column] = pd.to_datetime(df[datetime_column], format="%Y-%m-%d %H:%M:%S", errors="ignore")
        df[percent_columns] = df[percent_columns].astype(float)
        df[percent_columns] = df[percent_columns] * 0.01
        df[normal_columns] = df[normal_columns].astype(float)

        with open(os.path.join(cache_dir, "{}.pickle".format(path)), "wb") as f:
            pickle.dump(df, f)

    return df


def _init() -> None:
    np.random.seed(0)

    data_should_be = list([
        "MDS_Assignment2_Steelplates.xlsx",
        "MiningProcess_Flotation_Plant_Database.csv"
    ])

    if data_dir not in os.listdir(".") or \
        not np.all(np.isin(data_should_be, os.listdir(data_dir))):
        shutil.rmtree(data_dir, ignore_errors=True)

        ssl._create_default_https_context = ssl._create_unverified_context

        data_url = "https://www.techsol.cc/data/ntu/mds/mds_hw2_data.zip"
        zip_path = os.path.join(tempfile.gettempdir(), os.path.basename(data_url))
        urllib.request.urlretrieve(
            data_url,
            zip_path
        )

        os.makedirs(data_dir, exist_ok=True)
        with zipfile.ZipFile(
            zip_path,
            "r"
        ) as zip_file:
            zip_file.extractall(data_dir)
        os.remove(zip_path)


if __name__ == "__main__":
    _init()

    df = data_loader()
    print(df)
