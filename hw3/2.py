import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report


def _data_loader(target_shape=(300, 300), batch_size=128, random_seed=42):
    train_dir = os.path.join("casting_data", "casting_data", "train")
    test_dir = os.path.join("casting_data", "casting_data", "test")

    train_generator = ImageDataGenerator(rescale=1./255, validation_split=0.1)
    test_generator = ImageDataGenerator(rescale=1./255)

    training_generator = train_generator.flow_from_directory(
        directory=train_dir,
        subset="training",
        target_size=target_shape,
        color_mode="grayscale",
        classes=dict({
            "ok_front": 0,
            "def_front": 1
        }),
        class_mode="binary",
        batch_size=batch_size,
        seed=random_seed
    )
    validation_generator = train_generator.flow_from_directory(
        directory=train_dir,
        subset="validation",
        target_size=target_shape,
        color_mode="grayscale",
        classes=dict({
            "ok_front": 0,
            "def_front": 1
        }),
        class_mode="binary",
        batch_size=batch_size,
        seed=random_seed
    )
    testing_generator = test_generator.flow_from_directory(
        directory=test_dir,
        target_size=target_shape,
        color_mode="grayscale",
        classes=dict({
            "ok_front": 0,
            "def_front": 1
        }),
        class_mode="binary",
        batch_size=batch_size,
        seed=random_seed,
        shuffle=False
    )
    return (training_generator, validation_generator, testing_generator)


def _model_builder(input_shape, lr):
    model = Sequential()
    model.add(Conv2D(32, 3, activation="relu", padding="same", strides=2, input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=2, strides=2))
    model.add(Conv2D(64, 3, activation="relu", padding="same", strides=2))
    model.add(MaxPooling2D(pool_size=2, strides=2))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        loss="binary_crossentropy",
        optimizer=Adam(learning_rate=lr),
        metrics=["accuracy"]
    )
    model.summary()
    return model


target_shape = (300, 300)

training_generator, validation_generator, testing_generator = _data_loader(target_shape=target_shape)

model = _model_builder(target_shape+(1,), 0.001)
model.fit(
    training_generator,
    validation_data=validation_generator,
    epochs=30,
    verbose=1
)