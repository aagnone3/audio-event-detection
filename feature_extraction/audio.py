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


def log_filterbank_energy(fn, output_dir=os.getcwd(), sampling_rate=44100, duration_s=3,
                nfft=1024, n_freq_bins=64, frame_length_s=0.025, hop_length_s=0.010, force=False,
                mel_scale=False, **kwargs):

    duration = int(sampling_rate * duration_s)
    frame_length = int(sampling_rate * frame_length_s)
    hop_length = int(sampling_rate * hop_length_s)
    overlap_length = frame_length - hop_length
    window = np.hamming(frame_length)

    def extract(audio_fn):
        # Read and Resample the audio
        try:
            data, _ = librosa.core.load(audio_fn, sr=sampling_rate)
            data = normalize(data)
        except Exception as e:
            logging.exception(e)
            return None

        # ensure length
        if len(data) > duration:
            data = data[:duration]
        elif len(data) < duration:
            data = np.pad(data, (duration - len(data), ), mode='constant', constant_values=0)

        # spectrogram
        f, t, Sxx = sp.signal.spectrogram(
            data,
            fs=sampling_rate,
            window=window,
            nperseg=frame_length,
            noverlap=overlap_length,
            nfft=nfft
        )

        if mel_scale:
            # spectrogram -> log mel fb
            f_to_mel = filters.mel(sr=sampling_rate, n_fft=nfft, n_mels=n_freq_bins)
            Sxx = f_to_mel.dot(Sxx)

        Sxx = np.expand_dims(np.log(1e-8 + Sxx), axis=-1)

        return Sxx

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
        data = f.create_dataset('data', data=spectra)

    return out_fn


def log_filterbank_energy_sequence(fn, output_dir=os.getcwd(), sampling_rate=44100,
                nfft=1024, n_freq_bins=64, spectral_frame_length_s=1.0, spectral_hop_length_s=1.0,
                frame_length_s=0.025, hop_length_s=0.010, force=False, mel_scale=False, **kwargs):
    # NOTE: spectral_hop_length_s should be == spectral_frame_length_s for sequence models,
    # only > 0 for e.g. convolutional only

    frame_length = int(sampling_rate * frame_length_s)
    hop_length = int(sampling_rate * hop_length_s)
    spectral_frame_length = int(sampling_rate * spectral_frame_length_s)
    spectral_hop_length = int(sampling_rate * spectral_hop_length_s)
    overlap_length = frame_length - hop_length
    window = np.hamming(frame_length)
    spectral_chunk_size = int(sampling_rate * spectral_frame_length_s)
    n_frames_per_chunk = int(spectral_chunk_size / hop_length)
    spectral_hop_length_frames = int(spectral_hop_length / hop_length)

    def extract(audio_fn):
        # Read and Resample the audio
        try:
            data, _ = librosa.core.load(audio_fn, sr=sampling_rate)
            data = normalize(data)
        except Exception as e:
            logging.exception(e)
            return None

        n_spectral_chunks = int(np.ceil(float(len(data)) / spectral_chunk_size))
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

        if mel_scale:
            # spectrogram -> log mel fb
            f_to_mel = filters.mel(sr=sampling_rate, n_fft=nfft, n_mels=n_freq_bins)
            Sxx = f_to_mel.dot(Sxx)

        Sxx = np.expand_dims(np.log(1e-8 + Sxx), axis=-1)

        spectral_chunks = [
            zero_fill(Sxx[:, i:i + n_frames_per_chunk], (n_freq_bins, n_frames_per_chunk, 1))
            for i in range(0, Sxx.shape[1], spectral_hop_length_frames)
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
