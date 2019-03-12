import os
import h5py
import logging
from os import path
import numpy as np
import scipy as sp
from glob import glob
import multiprocessing as mp
from functools import partial

import librosa
from librosa.util import normalize
from librosa import filters


def indexed_sub_fn(fn, i):
    prefix, ext = path.splitext(fn)
    return prefix + str(i) + ext


def zero_fill(arr, target_shape):
    expansions = np.array([
        [0, target_shape[i] - arr.shape[i]]
        for i in range(len(target_shape))
    ])
    return np.pad(arr, expansions, mode='constant', constant_values=0)


def log_mel_fbe(fn, output_dir=os.getcwd(), sampling_rate=44100,
                nfft=1024, n_mels=64, spectral_frame_length_s=1.0, spectral_hop_length_s=0.250,
                frame_length_s=0.025, hop_length_s=0.010, force=False):

    frame_length = int(sampling_rate * frame_length_s)
    hop_length = int(sampling_rate * hop_length_s)
    spectral_frame_length = int(sampling_rate * spectral_frame_length_s)
    spectral_hop_length = int(sampling_rate * spectral_hop_length_s)
    f_to_mel = filters.mel(sr=sampling_rate, n_fft=nfft, n_mels=n_mels)
    overlap_length = frame_length - hop_length
    window = np.hamming(frame_length)
    spectral_chunk_size = int(sampling_rate * spectral_frame_length_s)
    n_frames_per_chunk = int(spectral_chunk_size / hop_length)

    def extract(audio_fn):
        # Read and Resample the audio
        try:
            data, _ = librosa.core.load(audio_fn, sr=sampling_rate)
            data = normalize(data)
        except Exception as e:
            logging.exception(e)
            return None

        n_spectral_chunks = (1 + len(data) // spectral_chunk_size)
        target_length = spectral_frame_length * n_spectral_chunks
        edge_pad_length = int(frame_length/2)
        # data = np.pad(data, ((edge_pad_length, edge_pad_length)), mode='constant', constant_values=0)

        # spectrogram
        f, t, Sxx = sp.signal.spectrogram(
            data,
            fs=sampling_rate,
            window=window,
            nperseg=frame_length,
            noverlap=overlap_length,
            nfft=nfft
        )

        # spectrogram -> log mel fb
        mel_spec = f_to_mel.dot(Sxx)
        log_mel_spec = np.log(1e-8 + mel_spec)
        log_mel_spec = np.expand_dims(log_mel_spec, axis=-1)
        spectral_chunk_hop_length = int(spectral_hop_length / hop_length)
        spectral_chunks = [
            zero_fill(log_mel_spec[:, i:i + n_frames_per_chunk], (n_mels, n_frames_per_chunk, 1))
            for i in range(0, log_mel_spec.shape[1], spectral_chunk_hop_length)
        ]

        return spectral_chunks

    # create the output directory if it does not exist
    if not path.exists(output_dir):
        os.mkdir(output_dir)

    # form the filename template, and check to see if features have already been extracted.
    # if they have, return the filenames without extracting again
    out_fn = path.join(output_dir, path.basename(fn) + '.hdf5')
    if path.exists(out_fn) and not force:
        return out_fn

    # extract features
    spectra = extract(fn)

    # indicate bad features if so
    if spectra is None:
        return "BAD_" + out_fn

    # write features to disk
    with h5py.File(out_fn, 'w') as f:
        meta = f.create_group('meta')
        t = meta.create_dataset('timestamps', data=np.zeros((len(spectra), 2)))
        data = f.create_dataset('data', data=np.zeros((len(spectra), *spectra[0].shape)))

        t_cur = 0
        for i, spec in enumerate(spectra):
            data[i] = spectra[i]
            t[i] = [t_cur, t_cur + spectral_frame_length_s]
            t_cur += spectral_frame_length_s

    return out_fn
