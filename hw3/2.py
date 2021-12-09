import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.utils import plot_model, to_categorical
from keras.preprocessing.image import ImageDataGenerator

# the data, shuffled and split between train and test sets
# from keras.datasets import mnist
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

cache_dir = ".cache"
os.makedirs(cache_dir, exist_ok=True)

if not os.path.exists(os.path.join(cache_dir, "casting_300.npz")):
    data_dir = os.path.join("data", "casting_data", "casting_data")

    train_dir = os.path.join(data_dir, "train")
    train_def_dir = os.path.join(train_dir, "def_front")
    train_ok_dir = os.path.join(train_dir, "ok_front")

    test_dir = os.path.join(data_dir, "test")

    X_train = list()
    y_train = list()

    for path in os.listdir(train_def_dir):
        X_train.append(cv2.imread(os.path.join(train_def_dir, path)))
        y_train.append(1)

    for path in os.listdir(train_ok_dir):
        X_train.append(cv2.imread(os.path.join(train_ok_dir, path)))
        y_train.append(0)

    X_train, X_test, y_train, y_test = train_test_split(
        np.array(X_train), np.array(y_train),
        test_size=0.1, random_state=42
    )

    np.savez(os.path.join(cache_dir, "casting_300"), X_train, X_test, y_train, y_test)

else:
    _cache = np.load(os.path.join(cache_dir, "casting_300.npz"))
    X_train, X_test, y_train, y_test = [_cache[k] for k in _cache]

datagen = ImageDataGenerator(
    rotation_range=15,
    zoom_range=0.2,
    shear_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.5,1.5]
)

print(X_train.shape)
# X_train = X_train.reshape(-1, 300, 300, 3)
# X_test = X_test.reshape(-1, 300, 300, 3)

datagen.fit(X_train)
datagen.fit(X_test)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=3, input_shape=(300, 300, 3), activation="relu", padding="same"))
model.add(MaxPool2D(pool_size=2))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='softmax'))
print(model.summary())

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

batch_size = 128
epochs = 10
history = model.fit(X_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(X_test, y_test))

# plt.imshow(X_train[1].reshape(28,28))
print('Prediction result: {}'.format(np.argmax(model.predict(X_train[1].reshape(1, 300, 300, 3)))))

# img = np.array(Image.open('test.png').resize((28,28)).convert('L')).astype('float32')
# img[img <= 50] = 0
# img /= 255.0

# plt.imshow(img)
# img = img.reshape(1, 28, 28, 1)
# print('Prediction result: {}'.format(np.argmax(model.predict(img))))