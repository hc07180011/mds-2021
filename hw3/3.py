import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

from typing import List

data_dir = "data"


def _data_loader(
    path: str = "Brent Oil Futures Historical Data.csv"
) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(data_dir, path))
    df["Mid"] = (df["High"] + df["Low"]) / 2
    df["Date"] = pd.to_datetime(df["Date"], format="%b %d, %Y")
    df = df.sort_values(by=["Date"])
    df.index = range(len(df))
    return df


def _preprocessing(df: pd.DataFrame, chunk_size=10) -> List[np.array]:
    prices = np.array(df["Mid"])
    scaler = MinMaxScaler()
    prices_scaled = scaler.fit_transform(prices.reshape(-1, 1))
    X = list()
    y = list()
    for i in range(chunk_size, len(prices), chunk_size):
        X.append(prices_scaled[i - chunk_size: i])
        y.append(prices_scaled[i: i + chunk_size])
    return np.array(X), np.array(y)


df = _data_loader()

chunk_size = 20

X, y = _preprocessing(df, chunk_size=chunk_size)

testing_split = 0.1
X_train = X[:-int(X.shape[0] * testing_split)]
X_test = X[-int(X.shape[0] * testing_split):]
y_train = y[:-int(X.shape[0] * testing_split)]
y_test = y[-int(X.shape[0] * testing_split):]

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

model = Sequential()
model.add(LSTM(units=64, return_sequences = True, input_shape=X.shape[1:]))
model.add(LSTM(units=32, return_sequences = True))
model.add(Dense(units=16, activation="relu"))
model.add(Flatten())
model.add(Dense(units=chunk_size, activation="sigmoid"))
model.compile(loss="mean_squared_error", optimizer=Adam(learning_rate=1e-5))
print(model.summary())

history = model.fit(
    X_train, y_train,
    epochs=1000,
    batch_size=16,
    validation_split=0.1,
)

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.legend(["loss", "validation loss"])
plt.savefig("3.png")
plt.close()

y_pred = model.predict(X_test)

plt.figure(figsize=(16, 8))
for i, y in enumerate(y_test):
    plt.plot(np.arange(i * chunk_size, (i + 1) * chunk_size), y, c="b")
for i, y in enumerate(y_pred):
    plt.plot(np.arange(i * chunk_size, (i + 1) * chunk_size), y, c="r")
plt.title("Best Test Predictions Over Time")
plt.xlabel("Time Series")
plt.ylabel("Mid Price")
plt.savefig("3_pred.png")
plt.close()