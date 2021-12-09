import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import f1_score, confusion_matrix


def _data_loader(target_shape=(300, 300), batch_size=128, random_seed=42):
    train_dir = os.path.join("data", "casting_data", "casting_data", "train")
    test_dir = os.path.join("data", "casting_data", "casting_data", "test")

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


def _f1_m(y_true, y_pred):
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


def _model_builder(input_shape, lr, conv_num):
    model = Sequential()
    if conv_num == 2:
        model.add(Conv2D(32, 3, activation="relu", padding="same", strides=2, input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=2, strides=2))
        model.add(Conv2D(64, 3, activation="relu", padding="same", strides=2))
        model.add(MaxPooling2D(pool_size=2, strides=2))
    else:
        model.add(Conv2D(64, 3, activation="relu", padding="same", strides=2, input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=2, strides=2))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        loss="binary_crossentropy",
        optimizer=Adam(learning_rate=lr),
        metrics=["accuracy", _f1_m]
    )
    return model


def _do_experiment(lr, conv_num, epochs=30):
    model = _model_builder(target_shape+(1,), lr, conv_num)
    history = model.fit(
        training_generator,
        validation_data=validation_generator,
        epochs=epochs,
        verbose=1,
        callbacks=[ModelCheckpoint(
            "model_{}_{}.h5".format(lr, conv_num),
            save_best_only=True,
            monitor="val__f1_m",
            mode="max"
        )]
    )
    np.save("{}_{}".format(lr, conv_num), history.history)


def _evaluation(lr, conv_num):
    history = np.load("{}_{}.npy".format(lr, conv_num), allow_pickle=True).tolist()
    plt.plot(history["loss"][1:])
    plt.plot(history["val_loss"][1:])
    plt.plot(history["_f1_m"][1:])
    plt.plot(history["val__f1_m"][1:])
    plt.legend(["loss", "validation loss", "f1", "validation f1"])
    plt.savefig("2_{}_{}.png".format(lr, conv_num))
    plt.close()

    model = load_model("model_{}_{}.h5".format(lr, conv_num), custom_objects={"_f1_m": _f1_m})
    model.evaluate(testing_generator)

    threshold = 0.5
    y_pred = (model.predict(testing_generator) > threshold).flatten()
    y_true = testing_generator.classes[testing_generator.index_array]

    print("===== lr = {}, conv_num = {} =====".format(lr, conv_num))
    print(f1_score(y_true, y_pred))
    print(confusion_matrix(y_true, y_pred))


target_shape = (300, 300)

training_generator, validation_generator, testing_generator = _data_loader(target_shape=target_shape)

learning_rate_list = list([0.1, 0.01, 0.001, 0.0001])

for lr in learning_rate_list:
    _do_experiment(lr, 2)
_do_experiment(0.001, 1)

for lr in learning_rate_list:
    _evaluation(lr, 2)
_evaluation(0.001, 1)