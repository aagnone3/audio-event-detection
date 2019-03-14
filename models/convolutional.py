import librosa
import numpy as np
import scipy
from keras import losses, optimizers
from keras.models import Sequential, Model
from keras.activations import relu, softmax
from keras.callbacks import (EarlyStopping, LearningRateScheduler,
                             ModelCheckpoint, TensorBoard, ReduceLROnPlateau)
from keras.layers import (Convolution1D, Dense, Dropout, GlobalAveragePooling1D,
                          GlobalMaxPool1D, GlobalMaxPool2D, Input, MaxPool1D,
                          MaxPool2D, MaxPool1D,
                          Flatten, Input, LeakyReLU,
                          BatchNormalization, Convolution2D, concatenate, Activation)
from keras.utils import Sequence, to_categorical
from keras import backend as K
from models.base import BaseModel


class CNN(BaseModel):

    defaults = {
        "learning_rate": 0.001
    }

    def __init__(self, input_dim, n_classes, **kwargs):
        super(CNN, self).__init__(**kwargs)

    def build_seq_model(self):
        self.model = Sequential([
            Convolution2D(32, (4, 10), padding="same", input_shape=self.input_dim),
            BatchNormalization(),
            Dropout(0.25),
            LeakyReLU(),
            MaxPool2D(),

            Convolution2D(64, (4, 10), padding="same"),
            BatchNormalization(),
            Dropout(0.25),
            LeakyReLU(),
            MaxPool2D(),

            Convolution2D(128, (4, 10), padding="same"),
            BatchNormalization(),
            Dropout(0.25),
            LeakyReLU(),
            MaxPool2D(),

            Convolution2D(256, (4, 10), padding="same"),
            BatchNormalization(),
            Dropout(0.25),
            LeakyReLU(),
            MaxPool2D(),

            Flatten(),
            Dense(128),
            BatchNormalization(),
            Dropout(0.25),
            LeakyReLU(),
            Dense(self.n_classes, activation=softmax)
        ])

        self.model.compile(
            optimizer=optimizers.Adam(self.learning_rate),
            loss=losses.categorical_crossentropy,
            metrics=['acc']
        )

    def build_model(self):
        inp = Input(shape=self.input_dim)
        x = Convolution2D(32, (4, 10), padding="same")(inp)
        x = BatchNormalization()(x)
        x = Dropout(0.25)(x)
        x = Activation("relu")(x)
        x = MaxPool2D((2, 10))(x)

        x = Convolution2D(64, (4, 10), padding="same")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.25)(x)
        x = Activation("relu")(x)
        x = MaxPool2D((4, 100))(x)

        # x = Convolution2D(128, (4, 10), padding="same")(x)
        # x = BatchNormalization()(x)
        # x = Dropout(0.25)(x)
        # x = Activation("relu")(x)
        # x = MaxPool2D((2, 10))(x)

        # x = Convolution2D(256, (4, 10), padding="same")(x)
        # x = BatchNormalization()(x)
        # x = Dropout(0.25)(x)
        # x = Activation("relu")(x)
        # x = MaxPool2D((2, 10))(x)

        x = Flatten()(x)
        # x = Dense(128)(x)
        # x = BatchNormalization()(x)
        # x = Activation("relu")(x)
        out = Dense(self.n_classes, activation=softmax)(x)

        self.model = Model(inputs=inp, outputs=out)
        opt = optimizers.Adam(self.learning_rate)

        self.model.compile(
            optimizer=opt,
            loss=losses.categorical_crossentropy,
            metrics=['acc']
        )
