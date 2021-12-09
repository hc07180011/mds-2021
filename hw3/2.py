import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from keras import backend as K
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import mobilenet


cache_dir = ".cache"
os.makedirs(cache_dir, exist_ok=True)

if not os.path.exists(os.path.join(cache_dir, "casting_300.npz")):
    data_dir = os.path.join("data", "casting_data", "casting_data")

    train_dir = os.path.join(data_dir, "train")
    train_def_dir = os.path.join(train_dir, "def_front")
    train_ok_dir = os.path.join(train_dir, "ok_front")

    test_dir = os.path.join(data_dir, "test")
    test_def_dir = os.path.join(test_dir, "def_front")
    test_ok_dir = os.path.join(test_dir, "ok_front")

    X_train = list()
    y_train = list()

    for path in os.listdir(train_def_dir):
        img = cv2.imread(os.path.join(train_def_dir, path))
        X_train.append(img)
        y_train.append(1)

    for path in os.listdir(train_ok_dir):
        img = cv2.imread(os.path.join(train_ok_dir, path))
        X_train.append(img)
        y_train.append(0)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    X_test = list()
    y_test = list()

    for path in os.listdir(test_def_dir):
        img = cv2.imread(os.path.join(test_def_dir, path))
        X_test.append(img)
        y_test.append(1)

    for path in os.listdir(test_ok_dir):
        img = cv2.imread(os.path.join(test_ok_dir, path))
        X_test.append(img)
        y_test.append(0)

    X_test = np.array(X_test)
    y_test = np.array(y_test)

    np.savez(os.path.join(cache_dir, "casting_300"), X_train, y_train, X_test, y_test)

else:
    _cache = np.load(os.path.join(cache_dir, "casting_300.npz"))
    X_train, y_train, X_test, y_test = [_cache[k] for k in _cache]


def preprocess_X(X):
    X = X.astype("float32")
    X /= 255
    return X


X_train = preprocess_X(X_train)
X_test = preprocess_X(X_test)

model = Sequential()
model.add(mobilenet.MobileNet(
    weights="imagenet", input_shape=(X_train.shape[1:]), include_top=False
))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dense(1, activation="softmax"))
print(model.summary())


def f1_m(y_true, y_pred):
    def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy", f1_m]
)

batch_size = 128
epochs = 5

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.1,
    random_state=42
)

history = model.fit(
    X_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=(X_val, y_val)
)

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.legend(["loss", "validation loss"])
plt.savefig("loss.png")
plt.close()

model.evaluate(X_test, y_test)

y_pred = model.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(f1_score(y_test, y_pred))