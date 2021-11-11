import os
import sys
import ssl
import copy
import shutil
import pickle
import zipfile
import tempfile
import urllib.request

try:
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
    from sklearn.linear_model import Ridge, Lasso, ElasticNet
    from sklearn.model_selection import train_test_split
    from statsmodels.stats.outliers_influence import variance_inflation_factor
except ImportError:
    print("Please run: pip install -r requirements.txt")
    sys.exit(0)


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

    X_train, _, y_train, __ = train_test_split(
        sm.add_constant(df[list([
            col for col in df.columns if col not in list(["date", "% Iron Concentrate", "% Silica Concentrate"])
        ])]),
        df["% Silica Concentrate"],
        test_size=0.33, random_state=42
    )

    # model = sm.OLS(y_train, X_train).fit()

    # pvalue_results = dict({
    #     "name": np.array(list([
    #         item[0] for item in model.pvalues.items()
    #     ])),
    #     "pvalue": np.array(list([
    #         item[1] for item in model.pvalues.items()
    #     ]))
    # })
    # sorted_names = pvalue_results["name"][np.argsort(pvalue_results["pvalue"])]
    # sorted_pvalues = pvalue_results["pvalue"][np.argsort(pvalue_results["pvalue"])]
    # print(pd.DataFrame({
    #     "Variable name": sorted_names,
    #     "p-value": sorted_pvalues,
    # }, index=None))

    # keep_idxs = list([x for x in range(1, len(df.columns) - 2)])
    # maximum_rsquared = model.rsquared
    # while keep_idxs:
    #     print("Current r-squared = {:.8f}".format(maximum_rsquared))
    #     maximum_rsquared_idx = -1
    #     for i in keep_idxs:
    #         trial_idx = copy.deepcopy(keep_idxs)
    #         trial_idx.remove(i)

    #         X_train_tmp, _, y_train_tmp, __ = train_test_split(
    #             sm.add_constant(df.iloc[:, trial_idx]),
    #             pd.DataFrame(df.iloc[:, -1]),
    #             test_size=0.33, random_state=42
    #         )
    #         model = sm.OLS(
    #             y_train_tmp,
    #             X_train_tmp
    #         ).fit()
    #         print("Trying: {}, r-squared: {:.8f}, better: {}".format(trial_idx, model.rsquared, model.rsquared > maximum_rsquared))
    #         if model.rsquared > maximum_rsquared:
    #             maximum_rsquared = model.rsquared
    #             maximum_rsquared_idx = i
    #     if maximum_rsquared_idx == -1:
    #         print("Already the best feature set!")
    #         break
    #     else:
    #         print("Removing: {}".format(maximum_rsquared_idx))
    #         keep_idxs.remove(maximum_rsquared_idx)

    clf = Ridge()
    clf.fit(X_train, y_train)

    coef_results = dict({
        "name": np.array(list([
            name for name in df.columns[1: -2]
        ])),
        "coef": np.array(list([
            coef for coef in clf.coef_[1:]
        ]))
    })

    sorted_names = coef_results["name"][np.argsort(-np.abs(coef_results["coef"]))]
    sorted_pvalues = coef_results["coef"][np.argsort(-np.abs(coef_results["coef"]))]
    print(pd.DataFrame({
        "Variable name": sorted_names,
        "Coef": sorted_pvalues,
    }, index=None))

    clf = Lasso()
    clf.fit(X_train, y_train)

    coef_results = dict({
        "name": np.array(list([
            name for name in df.columns[1: -2]
        ])),
        "coef": np.array(list([
            coef for coef in clf.coef_[1:]
        ]))
    })
    sorted_names = coef_results["name"][np.argsort(-np.abs(coef_results["coef"]))]
    sorted_pvalues = coef_results["coef"][np.argsort(-np.abs(coef_results["coef"]))]
    print(pd.DataFrame({
        "Variable name": sorted_names,
        "Coef": sorted_pvalues,
    }, index=None))

    clf = ElasticNet()
    clf.fit(X_train, y_train)

    coef_results = dict({
        "name": np.array(list([
            name for name in df.columns[1: -2]
        ])),
        "coef": np.array(list([
            coef for coef in clf.coef_[1:]
        ]))
    })
    sorted_names = coef_results["name"][np.argsort(-np.abs(coef_results["coef"]))]
    sorted_pvalues = coef_results["coef"][np.argsort(-np.abs(coef_results["coef"]))]
    print(pd.DataFrame({
        "Variable name": sorted_names,
        "Coef": sorted_pvalues,
    }, index=None))

    VIF_results = np.array([variance_inflation_factor(
        X_train.values,
        i
    ) for i in range(X_train.shape[1])])
    VIF_thres = 5
    print(np.where(VIF_results > VIF_thres), VIF_results[VIF_results > VIF_thres])


    X_train, _, y_train, __ = train_test_split(
        sm.add_constant(df[list([
            col for col in df.columns if col not in list(["date", "% Silica Concentrate"])
        ])]),
        df["% Silica Concentrate"],
        test_size=0.33, random_state=42
    )

    model = sm.OLS(y_train, X_train).fit()
    print(model.rsquared)
