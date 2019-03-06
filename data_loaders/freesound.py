import os
from os import path
import numpy as np
import scipy as sp
import pandas as pd
import torch
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
from sklearn.preprocessing import StandardScaler

from keras.utils import to_categorical
from keras.utils import Sequence
from librosa.core import stft
from librosa.util import normalize
from librosa import filters
import librosa
import joblib

from feature_extraction.audio import log_mel_fbe
from data_loaders.base import BaseDataLoader


class FreesoundDataGenerator(BaseDataLoader):

    defaults = {
        "sampling_rate": 44100,
        "audio_duration": 2,
        "n_classes": 41,
        "nfft": 1024,
        "n_mels": 64,
        "frame_size_ms": 25,
        "hop_size_ms": 15,
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
        self.frame_length = int(self.sampling_rate * self.frame_size_ms / 1000.0)
        self.hop_length = int(self.sampling_rate * self.hop_size_ms / 1000.0)
        self.overlap_length = self.frame_length - self.hop_length
        self.n_time_frames = int(np.floor(self.audio_length / self.hop_length)) - 1
        # self.feature_dim = (self.n_mels, self.n_time_frames, 1)  # CNN
        self.feature_dim = (self.n_time_frames, self.n_mels, 1)  # vggish
        self.window = np.hanning(self.frame_length)

        # TODO handle preprocessing functions
        self.preprocessing_fn = preprocessing_fn
        self.on_epoch_end()
        self.scaler = joblib.load("etc/scaler.pkl")

    def __len__(self):
        # number of batches per epoch
        return int(np.floor(len(self.indices) / self.batch_size))

    def __getitem__(self, index):
        indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        return self.__data_generation(indices)

    def on_epoch_end(self):
        self.indices = np.arange(len(self.file_paths))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __data_generation(self, indices):
        cur_batch_size = len(indices)
        X = np.empty((cur_batch_size, *self.feature_dim))

        for i, index in enumerate(indices):
            # X[i] = np.load(self.file_paths[index])['spec']  # CNN
            # X[i,:,:,0] = self.scaler.transform(X[i,:,:,0].T).T  # CNN
            X[i] = np.swapaxes(np.load(self.file_paths[index])['spec'], 0, 1)  # vggish
            # import pdb; pdb.set_trace()
            X[i,:,:,0] = self.scaler.transform(X[i,:,:,0])

        y = to_categorical(self.labels[indices], num_classes=self.n_classes)

        return X, y


class RTFreesoundDataGenerator(BaseDataLoader):

    defaults = {
        "sampling_rate": 44100,
        "audio_duration": 2,
        "n_classes": 41,
        "nfft": 1024,
        "n_mels": 64,
        "frame_size_ms": 25,
        "hop_size_ms": 15,
        "shuffle": False,
        "batch_size": 64
    }

    def __init__(self, file_paths=None, labels=None,
                 preprocessing_fn=lambda x: x, **kwargs):

        super(RTFreesoundDataGenerator, self).__init__(**kwargs)
        self.file_paths = file_paths
        self.indices = np.arange(len(self.file_paths))
        self.labels = labels
        self.audio_length = self.sampling_rate * self.audio_duration
        self.frame_length = int(self.sampling_rate * self.frame_size_ms / 1000.0)
        self.hop_length = int(self.sampling_rate * self.hop_size_ms / 1000.0)
        self.overlap_length = self.frame_length - self.hop_length
        self.feature_dim = (self.n_mels, int(np.floor(self.audio_length/self.hop_length)) - 1, 1)
        self.window = np.hanning(self.frame_length)

        # TODO handle preprocessing functions
        self.preprocessing_fn = preprocessing_fn
        self.on_epoch_end()
        self.scaler = joblib.load("scaler.pkl")

    def __len__(self):
        # number of batches per epoch
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

        with mp.Pool(mp.cpu_count()) as pool:
            mp_func = partial(log_mel_fbe, return_values=True, output_dir=None, sampling_rate=self.sampling_rate,
                             nfft=self.nfft, n_mels=self.n_mels, duration=self.audio_duration,
                             frame_length=self.frame_length, hop_length=self.hop_length, window=self.window)
            features = pool.map_async(mp_func, self.file_paths[indices]).get()

        for i in range(cur_batch_size):
            X[i,:,:,0] = self.scaler.transform(features[i][:,:,0].T).T

        y = to_categorical(self.labels[indices], num_classes=self.n_classes)
        return X, y
