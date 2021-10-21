import os
import sys
import ssl
import shutil
import urllib
import zipfile
import tempfile

try:
    import scipy
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    from statsmodels.regression.linear_model import RegressionResultsWrapper
    from statsmodels.stats.outliers_influence import variance_inflation_factor
except ImportError:
    print("Please run: pip install -r requirements.txt")
    sys.exit(0)


def data_loader(
    path: str = os.path.join("data", "MDS_Assignment1_furnace.csv")
) -> pd.DataFrame:
    return pd.read_csv(path, delimiter=",")


def train_linear_regression(
    Y: pd.DataFrame,
    X: pd.DataFrame
) -> RegressionResultsWrapper:
    return sm.OLS(Y, X).fit()


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

    model = train_linear_regression(
        pd.DataFrame(df.iloc[:,-1]),
        sm.add_constant(df.iloc[:, :-1])
    )

    print("\n\n1. (35%) Linear Regression Analysis for Wine Quality")

    print("\n(a) (10%) Show the results of regression analysis as follows.")
    print(model.summary())

    print("\n(b) (5%) The fitting of the linear regression is a good idea? If yes, why? If no, why? What’s the possible reason of poor fitting?")
    print("R-squared = {}".format(model.rsquared))

    pvalue_threshold = 0.01
    pvalue_results = dict({
        "name": np.array(list([
            item[0] for item in model.pvalues.items()
        ])),
        "pvalue": np.array(list([
            item[1] for item in model.pvalues.items()
        ]))
    })

    sorted_names = pvalue_results["name"][np.argsort(pvalue_results["pvalue"])]
    sorted_pvalues = pvalue_results["pvalue"][np.argsort(pvalue_results["pvalue"])]

    print("\n(c) (5%) Based on the results, rank the independent variables by p-values and which one are statistically significant variables with p-values<0.01? (i.e. 重要變數挑選)")
    print(pd.DataFrame({
        "Variable name": sorted_names[sorted_pvalues < pvalue_threshold],
        "p-value": sorted_pvalues[sorted_pvalues < pvalue_threshold],
    }, index=None))

    print("\n(d) (15%) Testify the underlying assumptions of regression (1) Normality, (2) Independence, and (3) Homogeneity of Variance with respect to residual.")
    
    print("(1) Normality")
    print(scipy.stats.shapiro(model.resid))
    print(scipy.stats.normaltest(model.resid))
    # print(scipy.stats.chisquare(model.resid))
    print(scipy.stats.jarque_bera(model.resid))
    print(scipy.stats.kstest(model.resid, "norm"))

    print("(2) Independence")
    VIF_results = np.array([variance_inflation_factor(
        sm.add_constant(df.iloc[:, :-1]).values,
        i
    ) for i in range(df.shape[1])])
    VIF_thres = 5
    print(np.where(VIF_results > VIF_thres), VIF_results[VIF_results > VIF_thres])

    print("(3) Homogeneity")
    print(scipy.stats.levene(*np.transpose(df.iloc[:, :-1].to_numpy())))
    print(scipy.stats.bartlett(*np.transpose(df.iloc[:, :-1].to_numpy())))


if __name__ == "__main__":
    main()