import os
from os import path
import numpy as np
import scipy as sp
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

from keras.utils import to_categorical
from keras.utils import Sequence
from librosa.core import stft
from librosa.filters import mel
from librosa.feature import melspectrogram
import librosa
import joblib
import matplotlib.pyplot as plt
from numpy.fft import rfft
from librosa.display import specshow
from data_loaders.base import BaseDataLoader


class FreesoundDataGenerator(BaseDataLoader):

    defaults = {
        "sampling_rate": 44100,
        "audio_duration": 2,
        "n_classes": 41,
        "n_mfcc": 40,
        "shuffle": False,
        "batch_size": 64
    }

    def __init__(self, file_paths=None, labels=None,
                 preprocessing_fn=lambda x: x, **kwargs):

        super(FreesoundDataGenerator, self).__init__(**kwargs)
        self.file_paths = file_paths
        self.indices = np.arange(len(self.file_paths))
        self.labels = labels
        self.audio_length = self.sampling_rate * self.audio_duration
        self.feature_dim = (self.n_mfcc, 1 + int(np.floor(self.audio_length/512)), 1)

        # TODO handle preprocessing functions
        self.preprocessing_fn = preprocessing_fn
        self.on_epoch_end()

    def __len__(self):
        # number of batches per epoch
        print(len(self.indices), self.batch_size)
        return int(np.floor(len(self.indices) / self.batch_size))

    def __getitem__(self, index):
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        return self.__data_generation(indices)

    def on_epoch_end(self):
        self.indices = np.arange(len(self.file_paths))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __data_generation(self, indices):
        cur_batch_size = len(indices)
        X = np.empty((cur_batch_size, *self.feature_dim))
        y = np.empty((cur_batch_size, self.n_classes))

        input_length = self.audio_length
        scaler = StandardScaler()
        # for i, index in tqdm(enumerate(indices), total=cur_batch_size):
        for i, index in enumerate(indices):
            # Read and Resample the audio
            data, _ = librosa.core.load(self.file_paths[index], sr=self.sampling_rate,
                                        res_type='kaiser_fast')

            # Random offset / Padding
            if len(data) > input_length:
                max_offset = len(data) - input_length
                offset = np.random.randint(max_offset)
                data = data[offset:(input_length+offset)]
            else:
                if input_length > len(data):
                    max_offset = input_length - len(data)
                    offset = np.random.randint(max_offset)
                else:
                    offset = 0
                data = np.pad(data, (offset, input_length - len(data) - offset), "constant")

            data = librosa.feature.mfcc(data, sr=self.sampling_rate,
                                               n_mfcc=self.n_mfcc)
            scaler.partial_fit(data.T)
            data = np.expand_dims(data, axis=-1)

            X[i] = data
            y[i] = self.labels[i]

        for i in range(X.shape[0]):
            X[i,:,:,0] = scaler.transform(X[i,:,:,0].T).T
        return X, y
