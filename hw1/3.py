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

    from mlxtend.preprocessing import TransactionEncoder
    from mlxtend.frequent_patterns import apriori
    from mlxtend.frequent_patterns import association_rules
except ImportError:
    print("Please run: pip install -r requirements.txt")
    sys.exit(0)


def data_loader(
    path: str = os.path.join("data", "MDS_Assignment1_groceries.csv")
) -> pd.DataFrame:
    raw_data = list([x.split(",") for x in open(path, "r").read().split("\n")[:-1]])
    te = TransactionEncoder()
    te_ary = te.fit(raw_data).transform(raw_data)
    return pd.DataFrame(te_ary, columns=te.columns_)


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

    print("\n\n3. (35%) Association Rule- Market Basket Analysis")
    
    print("\n(1) (10%) How to handle the raw dataset via data preprocessing?")
    print(df)

    print("\n(2) (10%) What’s the top 5 association rules? Show the support, confidence, and lift to each specific rule, respectively?")

    frequent_itemset = apriori(df, min_support=0.001, use_colnames=True)
    rules = association_rules(
        frequent_itemset,
        min_threshold=0.15
    )

    print(rules.sort_values("support", ascending=False).head(5))
    print(rules.sort_values("confidence", ascending=False).head(5))
    print(rules.sort_values("lift", ascending=False).head(5))

    print("\n(3) (5%) Please provide/guess the “story” to interpret one of top-5 rules you are interested in.")
    
    print("\n(4) (10%) Give a visualization graph of your association rules.")

    plt.scatter(rules["support"], rules["confidence"], s=1)
    plt.xlabel("Support")
    plt.ylabel("Confidence")
    plt.savefig("3-4.png")
    plt.close()


if __name__ == "__main__":
    main()