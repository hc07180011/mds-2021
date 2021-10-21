import os
import sys
import ssl
import shutil
import urllib
import zipfile
import tempfile

try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
except ImportError:
    print("Please run: pip install -r requirements.txt")
    sys.exit(0)


def data_loader(
    path: str = os.path.join("data", "MDS_Assignment1_census.csv")
) -> pd.DataFrame:
    ret_df = pd.read_csv(path, delimiter=",", header=None)
    ret_df.columns = list([
        "age", "workclass", "fnlwgt",
        "education", "education-num",
        "marital-status", "occupation",
        "relationship", "race", "sex",
        "capital-gain", "capital-loss",
        "hours-per-week", "native-country",
        "class"
    ])
    for key in ret_df.columns:
        if ret_df.dtypes[key] == np.int64:
            ret_df = ret_df.astype(dict({key: np.float64}))
    return ret_df


def _init() -> None:
    np.random.seed(0)

    csv_should_be = list([
        "MDS_Assignment1_census.csv",
        "MDS_Assignment1_furnace.csv",
        "MDS_Assignment1_groceries.csv"
    ])

    if "data" not in os.listdir(".") or \
        not np.all(np.isin(csv_should_be, os.listdir("data"))):
        shutil.rmtree("data", ignore_errors=True)

        ssl._create_default_https_context = ssl._create_unverified_context

        data_url = "https://www.techsol.cc/data/ntu/mds/mds_hw1_data.zip"
        zip_path = os.path.join(tempfile.gettempdir(), os.path.basename(data_url))
        urllib.request.urlretrieve(
            data_url,
            zip_path
        )

        os.makedirs("data", exist_ok=True)
        with zipfile.ZipFile(
            zip_path,
            "r"
        ) as zip_file:
            zip_file.extractall("data")
        os.remove(zip_path)


def main() -> None:
    _init()

    df = data_loader()
    missing_value_format = " ?"
    
    print("\n\n2. (30%) Data Preprocessing and Generalized Linear Model (GLM)/Logistic Regression")

    numeric_cols = list([
        key for key in df.columns if df.dtypes[key] == np.float64
    ])
    categorical_cols = list([
        key for key in df.columns if df.dtypes[key] != np.float64
    ])

    print("\n(1) (5%) Provide the descriptive statistics. (i.e. exploratory data analysis, EDA) Eg. mean, variance, data distribution, # of missing value, # of outlier, etc.")
    print(df[numeric_cols].mean(axis=0))
    print(df[numeric_cols].var(axis=0))

    fig, axs = plt.subplots(2, 3, figsize=(18, 8))
    for i, key in enumerate(numeric_cols):
        axs[i // 3][i % 3].hist(df[key])
        axs[i // 3][i % 3].set_xlabel("{}".format(key))
        axs[i // 3][i % 3].set_ylabel("Counts")
    fig.savefig("2-1.png".format(key))
    plt.close()

    print(np.sum(df.to_numpy() == missing_value_format))

    zscore_outlier_thres = 3.0
    print(np.sum([
        np.sum(np.abs((df[key] - df[key].mean()) / df[key].std()) > zscore_outlier_thres)
        for key in numeric_cols
    ]))

    print("\n(2) (10%) How to identify the outlier? How to impute the missing value?")

    for key in numeric_cols:
        print("For {}:".format(key))
        print(df[key][np.abs((df[key] - df[key].mean()) / df[key].std()) > zscore_outlier_thres])

    for key in df.columns:
        if df.dtypes[key] != np.float64:
            values, counts = np.unique(df[key].to_numpy(), return_counts=True)
            imputing_value = (
                values[np.argsort(counts)[-1]]
                if values[np.argsort(counts)[-1]] != missing_value_format
                else values[np.argsort(counts)[-2]]
            )
            df[key] = df[key].replace(missing_value_format, imputing_value)

    print("\n(3) (5%) How to transform the categorical variable to dummy variable?")

    df = pd.get_dummies(
        df,
        columns=categorical_cols,
        drop_first=True
    )
    print(df)

    print("\n(4) (5%) How to “randomly” split the dataset into training dataset and testing dataset (eg. 80% vs. 20%)?")

    X_train, X_test, y_train, y_test = train_test_split(
        df.iloc[:, :-1],
        df.iloc[:, -1],
        test_size=0.2,
        random_state=np.random.randint(2**32 - 1)
    )
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    print("\n(5) (5%) Please use the Generalized Linear Model (GLM) (OR Logistic Regression) to predict the “Class” in the testing dataset.")

    model = LogisticRegression().fit(X_train, y_train)
    print(model.score(X_test, y_test))


if __name__ == "__main__":
    main()