import os
import sys
import ssl
import shutil
import zipfile
import tempfile
import subprocess
import urllib.request

try:
    import numpy as np
    import pandas as pd
    from sklearn import tree
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_validate
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, make_scorer, confusion_matrix
    from sklearn.utils import resample
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

    import matplotlib.pyplot as plt
except ImportError:
    print("Please run: pip install -r requirements.txt")
    sys.exit(0)


data_dir = "data"


def data_loader(
    path: str = "MDS_Assignment3_Steelplates.xlsx"
) -> pd.DataFrame:
    df_columnn_names = pd.read_excel(os.path.join(data_dir, path), sheet_name=0, header=None)
    df = pd.read_excel(os.path.join(data_dir, path), sheet_name=1, header=None)
    df.columns = df_columnn_names.to_numpy().flatten().tolist()
    return df


def _init() -> None:
    np.random.seed(0)

    data_should_be = list([
        "MDS_Assignment3_Steelplates.xlsx"
    ])

    if data_dir not in os.listdir(".") or \
        not np.all(np.isin(data_should_be, os.listdir(data_dir))):
        shutil.rmtree(data_dir, ignore_errors=True)

        ssl._create_default_https_context = ssl._create_unverified_context

        data_url = "https://www.techsol.cc/data/ntu/mds/mds_hw3_data.zip"
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


def _do_experiment(
        df: pd.DataFrame,
        classifier1: tree.DecisionTreeClassifier,
        classifier2: RandomForestClassifier,
        classifier3: GradientBoostingClassifier,
        code: str
    ) -> None:

    print("===== {} =====".format(code))

    X_train = df.iloc[:, :-num_labels]
    X_train = StandardScaler().fit_transform(X_train)

    y_train = np.argmax(df.iloc[:, -num_labels:].to_numpy(), axis=1)

    max_idx, max_f1 = -1, 0.0
    accuracy_record, f1_score_record, roc_auc_record = list(), list(), list()
    accuracy_record.append(np.zeros(10))
    f1_score_record.append(np.zeros(10))
    roc_auc_record.append(np.zeros(10))

    range_ = range(1, 33) if classifier1 else range(10, 110, 10) if classifier2 else [0.001, 0.01, 0.1, 1]
    for i in range_:
        results = cross_validate(
            classifier1(max_depth=i, random_state=42) if classifier1 \
                else classifier2(n_estimators=i, random_state=42) if classifier2 \
                    else classifier3(learning_rate=i, random_state=42),
            X_train,
            y_train,
            scoring=dict({
                "accuracy": make_scorer(accuracy_score),
                "f1": make_scorer(f1_score, average="macro"),
                "roc_auc": make_scorer(roc_auc_score, multi_class="ovr", needs_proba=True)
            }),
            cv=10
        )
        accuracy_record.append(results["test_accuracy"])
        f1_score_record.append(results["test_f1"])
        roc_auc_record.append(results["test_roc_auc"])
        if np.mean(results["test_f1"]) > max_f1:
            max_idx = i
            max_f1 = np.mean(results["test_f1"])

    print("Best index is: {}, f1-score = {}".format(max_idx, max_f1))

    x_label = "Max Depth" if classifier1 else "N Estimators" if classifier2 else "Learning Rate"

    plt.plot([0, ] + list(range_), [np.mean(x) for x in accuracy_record])
    plt.xlabel(x_label)
    plt.ylabel("Accuracy")
    plt.savefig("1-{}-acc.png".format(code))
    plt.close()

    plt.plot([0, ] + list(range_), [np.mean(x) for x in f1_score_record])
    plt.xlabel(x_label)
    plt.ylabel("F1 Score")
    plt.savefig("1-{}-f1.png".format(code))
    plt.close()

    plt.plot([0, ] + list(range_), [np.mean(x) for x in roc_auc_record])
    plt.xlabel(x_label)
    plt.ylabel("ROC AUC Score")
    plt.savefig("1-{}-roc.png".format(code))
    plt.close()

    X_train, X_test, y_train, y_test = train_test_split(
        X_train, y_train,
        test_size=0.1,
        random_state=42
    )

    clf = classifier1(max_depth=max_idx, random_state=42) if classifier1 \
        else classifier2(n_estimators=max_idx, random_state=42) if classifier2 \
            else classifier3(learning_rate=max_idx, random_state=42)
    clf.fit(X_train, y_train)

    if classifier1:
        with open(os.path.join(tempfile.gettempdir(), "tree.dot"), "w") as f:
            _ = tree.export_graphviz(clf, out_file=f)
        p = subprocess.Popen([
            "dot", "-T", "png",
            "{}".format(os.path.join(tempfile.gettempdir(), "tree.dot")),
            "-o", "1-{}-tree.png".format(code)
        ])
        p.wait()
        os.remove(os.path.join(tempfile.gettempdir(), "tree.dot"))

    y_true = y_test
    y_pred = clf.predict(X_test)
    print("Final accuracy = {}".format(accuracy_score(y_true, y_pred)))
    print("Final f1-score = {}".format(f1_score(y_true, y_pred, average="macro")))
    print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    _init()

    df = data_loader()
    num_labels = 7

    # a
    print(df.describe())

    # b
    print(np.all(np.bitwise_xor(df["TypeOfSteel_A300"], df["TypeOfSteel_A400"])))
    prepared_df = df.drop(["TypeOfSteel_A400"], axis=1)

    # c
    _do_experiment(prepared_df, tree.DecisionTreeClassifier, None, None, "c")

    # d
    X_train = prepared_df.iloc[:, :-num_labels]
    X_train = StandardScaler().fit_transform(X_train)

    y_train = np.argmax(prepared_df.iloc[:, -num_labels:].to_numpy(), axis=1)

    for i in range(num_labels):
        plt.bar(i, len(np.where(y_train == i)[0]))
    plt.title("Count(s) of Every Class")
    plt.xticks(np.arange(num_labels).tolist(), df.columns[-num_labels:])
    plt.ylabel("Count(s)")
    plt.savefig("1-d.png")
    plt.close()

    balanced_df = pd.DataFrame()
    for i in range(num_labels):
        balanced_df = pd.concat([
            balanced_df,
            resample(df.iloc[np.where(y_train == i)[0], :], replace=True, n_samples=673, random_state=42)
        ])

    # e
    _do_experiment(balanced_df, tree.DecisionTreeClassifier, None, None, "e")

    # g
    _do_experiment(prepared_df, None, RandomForestClassifier, None, "g1")
    _do_experiment(balanced_df, None, RandomForestClassifier, None, "g2")

    # h
    _do_experiment(prepared_df, None, None, GradientBoostingClassifier, "h1")
    _do_experiment(balanced_df, None, None, GradientBoostingClassifier, "h2")