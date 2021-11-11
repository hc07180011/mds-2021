import os
import sys
import ssl
import shutil
import zipfile
import tempfile
import urllib.request

try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix
    from sklearn.decomposition import PCA
except ImportError:
    print("Please run: pip install -r requirements.txt")
    sys.exit(0)


data_dir = "data"
cache_dir = ".cache"


def data_loader(
    path: str = "MDS_Assignment2_Steelplates.xlsx"
) -> pd.DataFrame:
    return pd.read_excel(
        os.path.join(data_dir, path), 1,
        names=pd.read_excel(os.path.join(data_dir, path), 0, header=None)[0].values,
        header=None
    )


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

    X_train, X_test, y_train, y_test = train_test_split(
        df[df.columns[:27]],
        df["Bumps"],
        test_size=0.33, random_state=42
    )

    clf = LogisticRegression(max_iter=128)
    clf.fit(X_train, y_train)
    print("Accuracy: {}",format(clf.score(X_test, y_test)))

    y_pred = clf.predict(X_test)
    print(confusion_matrix(y_test, y_pred))

    pca = PCA()
    X_train_pca = pca.fit_transform(
        X_train.drop(list(["TypeOfSteel_A300", "TypeOfSteel_A400"]), axis=1)
    )
    print("Eigenvalues = {}".format(pca.explained_variance_))
    print("Eigenvectors = {}".format(pca.components_))
    print("Variance Ratio = {}".format(pca.explained_variance_ratio_))

    plt.plot(pca.explained_variance_, c="b")
    plt.scatter(np.arange(X_train.shape[1] - 2), pca.explained_variance_, c="b")
    plt.xlabel("# of Features")
    plt.ylabel("Explained Variance")
    plt.title("Explained Variace of Each Feature under PCA")
    plt.savefig("2.png")

    X_test_pca = pca.fit_transform(
        X_test.drop(list(["TypeOfSteel_A300", "TypeOfSteel_A400"]), axis=1)
    )

    clf = LogisticRegression(max_iter=128)
    clf.fit(X_train_pca[:, :2], y_train)
    print("Accuracy: {}".format(clf.score(X_test_pca[:, :2], y_test)))

    y_pred = clf.predict(X_test_pca[:, :2])
    print(confusion_matrix(y_test, y_pred))
