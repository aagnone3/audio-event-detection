import librosa
import numpy as np
import scipy
from keras import losses, optimizers
from keras import backend as K
from keras.models import Sequential, Model
from keras.activations import relu, softmax
from keras.callbacks import (EarlyStopping, LearningRateScheduler,
                             ModelCheckpoint, TensorBoard, ReduceLROnPlateau)
from keras.layers import (Convolution1D, Dense, Dropout, GlobalAveragePooling1D,
                          GlobalMaxPool1D, GlobalMaxPool2D, Input, MaxPool1D,
                          MaxPool2D, MaxPool1D, Lambda,
                          Flatten, Input, LeakyReLU, Conv1D, TimeDistributed, Activation,
                          Bidirectional, GRU, LSTM,
                          BatchNormalization, Convolution2D, concatenate, Activation)
from keras.utils import Sequence, to_categorical
from models.base import BaseModel


def ctc_loss(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def add_ctc_loss(softmax_layer):
    labels = Input(name='labels', shape=(None,), dtype='float32')
    input_lengths = Input(name='example_lengths', shape=(1,), dtype='int64')
    label_lengths = Input(name='label_lengths', shape=(1,), dtype='int64')
    output_lengths = Lambda(softmax_layer.output_length)(input_lengths)
    losses = Lambda(ctc_loss, output_shape=(1,), name='ctc')(
        [softmax_layer.output, labels, output_lengths, label_lengths]
    )
    return Model(
        inputs=[softmax_layer.input, labels, input_lengths, label_lengths],
        outputs=losses
    )


def cnn_rnn_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    """ Build a recurrent + convolutional network for speech
    """
    # Main acoustic input
    input_data = Input(name='input', shape=(None, input_dim))

    # Add convolutional layer with BN and ReLU
    conv_1d = Conv1D(filters, kernel_size,
                     strides=conv_stride,
                     padding=conv_border_mode,
                     name='conv1d')(input_data)
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    bn_cnn = Activation('relu')(bn_cnn)

    # Add a recurrent layer with BN and ReLU
    simp_rnn = LSTM(units, dropout=0.25,
        return_sequences=True, implementation=2, name='rnn')(bn_cnn)
    bn_rnn = BatchNormalization(name='bn_cnn_rnn')(simp_rnn)
    bn_rnn = Activation('relu')(bn_rnn)

    # Add a TimeDistributed(Dense(output_dim)) layer with softmax activation
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    y_pred = Activation('softmax', name='softmax')(time_dense)

    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def cnn_output_length(input_length, filter_size, border_mode, stride,
                       dilation=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    return (output_length + stride - 1) // stride


class CRNN(BaseModel):

    defaults = {
        "learning_rate": 0.001,
        "conv_stride": 1,
        "filter_size": 14,
        "rec_layers": 2,
        "rec_layer_size": 100,
        "rec_dropout": 0.25,
        "conv_border_mode": "valid",
        "conv_filters": 16
    }

    def __init__(self, input_dim, n_classes, **kwargs):
        super(CRNN, self).__init__(input_dim, n_classes, **kwargs)

    def build_model(self):
        # Main acoustic input
        input_data = Input(name='input', shape=(None, self.input_dim))

        # Add convolutional layer with BN and ReLU
        x = Conv1D(self.conv_filters, self.filter_size, strides=self.conv_stride, padding=self.conv_border_mode)(input_data)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        for i in range(self.rec_layers):
            x = Bidirectional(LSTM(self.rec_layer_size, dropout=self.rec_dropout, return_sequences=True, implementation=2))(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

        # TimeDistributed Dense layer with softmax activation
        x = TimeDistributed(Dense(self.n_classes))(x)
        y_pred = Activation('softmax', name='softmax')(x)

        # Specify the model
        self.model = Model(inputs=input_data, outputs=y_pred)
        self.model.output_length = lambda x: cnn_output_length(
            x, self.filter_size, self.conv_border_mode, self.conv_stride)
        print(self.model.summary())

        self.model = add_ctc_loss(self.model)
        self.model.compile(
            optimizer=optimizers.Adam(self.learning_rate),
            loss={'ctc': lambda y_true, y_pred: y_pred},
            # metrics=['acc']
        )
