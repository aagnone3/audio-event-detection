import os
import h5py
from os import path
import numpy as np
import scipy as sp
import pandas as pd
import torch
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

from keras.utils import to_categorical
from keras.utils import Sequence
from librosa.core import stft
from librosa.util import normalize
from librosa import filters
import librosa
from python_speech_features import mfcc
from collections import defaultdict

from feature_extraction.audio import log_mel_fbe
from data_loaders.base import BaseDataLoader


class DCASEDataGenerator(BaseDataLoader):

    defaults = {
        "sampling_rate": 44100,
        "audio_duration": 0.960,
        "n_classes": None,
        "nfft": 2048,
        "n_mels": 64,
        "frame_size_ms": 25,
        "hop_size_ms": 10,
        "shuffle": False
    }

    def __init__(self, mode="train", unique_labels_fn=None, preprocessing_fn=lambda x: x, **kwargs):
        super(DCASEDataGenerator, self).__init__(**kwargs)
        self.mode = mode
        mode_configs = kwargs.get(mode)
        if not mode_configs:
            raise ValueError("Expected key '{}' not found in data_loader configuration.".format(mode))
        self.meta = pd.read_csv(mode_configs["meta_file"])
        self.batch_size = mode_configs.get('batch_size', 32)
        self.feature_dir = mode_configs.get('feature_dir', 'features')
        self.file_paths = self.meta['filename'].map(lambda fn: path.join(self.feature_dir, fn + ".hdf5")).values
        self.indices = np.arange(len(self.file_paths))
        self.labels = self.meta["event_labels"].values
        
        # audio extraction parameters
        self.audio_length = self.sampling_rate * self.audio_duration
        self.frame_length = int(self.sampling_rate * self.frame_size_ms / 1000.0)
        self.hop_length = int(self.sampling_rate * self.hop_size_ms / 1000.0)
        self.overlap_length = self.frame_length - self.hop_length
        self.n_time_frames = int(np.floor(self.audio_length / self.hop_length)) - 1
        self.feature_dim = self.n_mels
        self.window = np.hanning(self.frame_length)

        # load label encoder and encode labels
        self.label_map = defaultdict(lambda: -1)
        with open(unique_labels_fn, 'r') as fp:
            self.unique_labels = np.unique([line.strip('\n') for line in fp.readlines()])
        for i, label in enumerate(self.unique_labels):
            self.label_map[label] = i
        self.labels = self.make_labels(self.label_map, self.labels, self.n_classes)

        # TODO handle preprocessing functions
        self.preprocessing_fn = preprocessing_fn
        self.on_epoch_end()

    def __len__(self):
        # number of batches per epoch
        return int(np.floor(len(self.indices) / self.batch_size))

    def __getitem__(self, index):
        indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        return self.__data_generation(indices)
    
    def make_labels(self, label_map, raw_labels, n_classes):
        """
        e.g. a,b -> 0101
        """
        labels = list()
        for label in raw_labels:
            if ',' in label:
                label_one_hots = to_categorical([label_map[_label] for _label in label.split(',')], num_classes=n_classes)
                labels.append(np.array(label_one_hots, dtype=int).sum(axis=0))
            else:
                labels.append(to_categorical(label_map[label], num_classes=self.n_classes))
        return np.array(labels)
                    

    def on_epoch_end(self):
        self.indices = np.arange(len(self.file_paths))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __data_generation(self, indices):
        cur_batch_size = len(indices)

        example_lengths = np.empty((len(indices), ), dtype=int)
        label_lengths = np.ones((len(indices), ), dtype=int)
        max_length = -1
        data = list()
        for i, index in enumerate(indices):
            with h5py.File(self.file_paths[index], 'r') as f:
                data.append(np.reshape(f['data'], (-1, self.n_mels)))
                example_lengths[i] = data[-1].shape[0]
                if example_lengths[i] > max_length:
                    max_length = example_lengths[i]
                
        X = np.empty((cur_batch_size, max_length, self.feature_dim))
        for i, index in enumerate(indices):
            X[i][:data[i].shape[0]][:] = data[i]

        y = self.labels[indices]
        
        inputs = {
            'input': X,
            'labels': y,
            'label_lengths': label_lengths,
            'example_lengths': example_lengths
        }
        outputs = {
            'ctc': np.zeros([cur_batch_size])
        }
        return inputs, outputs
