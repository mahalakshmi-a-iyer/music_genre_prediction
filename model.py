"""
CNN model architecture for music genre classification.

Architecture: 5 convolutional blocks with BatchNorm, followed by a dense classifier.
Total parameters: ~11.4M
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import config


def build_model(input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, 1),
                num_classes=config.NUM_CLASSES):
    """
    Build and compile the CNN model.

    Architecture:
        5x [Conv2D -> BatchNorm -> ReLU -> MaxPool2D -> Dropout(0.30)]
        Flatten -> Dense(1200) -> BatchNorm -> Dropout(0.45) -> Dense(10, softmax)

    Returns a compiled Keras model.
    """
    model = keras.Sequential()
    model.add(layers.Input(shape=input_shape))

    # 5 convolutional blocks
    for filters in config.CONV_FILTERS:
        model.add(layers.Conv2D(filters, config.KERNEL_SIZE,
                                padding="same", activation="relu"))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D(pool_size=config.POOL_SIZE))
        model.add(layers.Dropout(config.CONV_DROPOUT))

    # Classifier head
    model.add(layers.Flatten())
    model.add(layers.Dense(config.DENSE_UNITS, activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(config.DENSE_DROPOUT))
    model.add(layers.Dense(num_classes, activation="softmax"))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


if __name__ == "__main__":
    model = build_model()
    model.summary()
