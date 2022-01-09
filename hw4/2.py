import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import variation
from sklearn.ensemble import RandomForestRegressor


def _data_loader() -> pd.DataFrame:
    column_names = \
        list(("unit number", "time in cycles")) + \
        list([
            "operational setting {}".format(i + 1)
            for i in range(3)
        ]) + \
        list([
            "sensor measurement {}".format(i + 1)
            for i in range(21)
        ])
    return pd.read_csv("train.txt", delimiter=" ", names=column_names, index_col=False)


def _get_ma(data: np.array, window_size: int = 10) -> np.array:
    ret = list()
    for i in range(data.shape[0] - window_size + 1):
        ret.append(np.mean(data[i:(i + window_size)]))
    for _ in range(window_size - 1):
        ret.append("-")
    return ret


def _get_mv(data: np.array, window_size: int = 10) -> np.array:
    ret = list()
    for i in range(data.shape[0] - window_size + 1):
        ret.append(np.var(data[i:(i + window_size)]))
    for _ in range(window_size - 1):
        ret.append("-")
    return ret


def _get_mp(data: np.array, window_size: int = 10) -> np.array:
    ret = list()
    for i in range(data.shape[0] - window_size + 1):
        ret.append(np.max(data[i:(i + window_size)]))
    for _ in range(window_size - 1):
        ret.append("-")
    return ret


def _main() -> None:
    df = _data_loader()
    df = df[df["unit number"] == 1]

    rul_list = list()
    for i in range(len(df)):
        rul_list.append(np.max(df["time in cycles"]) - i - 1)
    df["RUL"] = rul_list

    print(df)

    variation_list = list()
    for sensor in list([
        "sensor measurement {}".format(i + 1)
        for i in range(21)
    ]):
        variation_list.append(variation(df[sensor]))

    print(np.argmax(variation_list), np.argmin(variation_list))

    for sensor in list([
        "sensor measurement {}".format(i + 1)
        for i in range(21)
    ]):
        df["{}_MA".format(sensor)] = (_get_ma(
            df[sensor]
        ))
        df["{}_MV".format(sensor)] = (_get_mv(
            df[sensor]
        ))
        df["{}_MP".format(sensor)] = (_get_mp(
            df[sensor]
        ))

    print(df)

    df["RUL_MA"] = _get_ma(df["RUL"])

    print(df.iloc[:-10, 27:].astype(float).corr()[-1:].T.sort_values(
        by=["RUL_MA"], key=lambda x: -np.abs(x)
    ).head(11))

    regr = RandomForestRegressor(random_state=0)
    regr.fit(
        df.iloc[:-10, 27:].astype(float).iloc[:, :-1],
        df.iloc[:-10, 27:].astype(float).iloc[:, -1:]
    )

    importance_ranking = np.argsort(-regr.feature_importances_)

    figure, axes = plt.subplots(5, 2)
    for i in range(10):
        print(
            df.columns[27 + importance_ranking[i]],
            regr.feature_importances_[importance_ranking[i]]
        )
        axes[i // 2][i % 2].plot(
            df[df.columns[27 + importance_ranking[i]]].to_numpy()[:-10]
        )
        axes[i // 2][i % 2].set_title(
            df.columns[27 + importance_ranking[i]]
        )
    plt.tight_layout()
    plt.savefig("2-e.png")


if __name__ == "__main__":
    _main()