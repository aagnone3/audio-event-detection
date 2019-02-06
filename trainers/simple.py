import os
import numpy as np
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau, ProgbarLogger

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

from utils import factory
from trainers.base import BaseTrain


class SimpleTrainer(BaseTrain):

    defaults = {
        "num_epochs": 50,
        "validation_split": 0.20,
        "verbose_training": False
    }

    def __init__(self, description, model, callbacks_config, **kwargs):
        super(SimpleTrainer, self).__init__(description, model, **kwargs)
        self.callbacks_config = callbacks_config
        self.loss = []
        self.acc = []
        self.val_loss = []
        self.val_acc = []
        self.best_model_fn = os.path.join(
            self.callbacks_config["checkpoint_dir"],
            "{}.hdf5".format(self.description)
        )

        self.callbacks = []
        self.init_callbacks()

    def init_callbacks(self):
        self.callbacks.append(
            ModelCheckpoint(
                filepath=self.best_model_fn,
                **self.callbacks_config["ModelCheckpoint"]
            )
        )
        self.callbacks.append(
            EarlyStopping(
                monitor="val_loss",
                patience=10
            )
        )
        self.callbacks.append(
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.1,
                patience=10,
                min_lr=int(1e-6)
            )
        )
        self.callbacks.append(
            TensorBoard(
                log_dir=self.callbacks_config["tensorboard_log_dir"],
                write_graph=self.callbacks_config["tensorboard_write_graph"],
            )
        )

    def train(self, training_data_generator, validation_data_generator):
        history = self.model.fit_generator(
            generator=training_data_generator,
            validation_data=validation_data_generator,
            epochs=self.num_epochs,
            verbose=self.verbose_training,
            callbacks=self.callbacks,
        )

        self.loss.extend(history.history['loss'])
        self.acc.extend(history.history['acc'])
        self.val_loss.extend(history.history['val_loss'])
        self.val_acc.extend(history.history['val_acc'])

    def test(self, data_generator):
        # load the weights for the best model
        self.model.load_weights(self.best_model_fn)

        # evaluate that model with the given data generator
        return self.model.evaluate_generator(
            generator=data_generator,
            verbose=self.verbose_training
        )
