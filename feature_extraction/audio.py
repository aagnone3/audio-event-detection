import os
from os import path
import numpy as np
import scipy as sp
import multiprocessing as mp
from functools import partial

import librosa
from librosa.util import normalize
from librosa import filters


def zero_fill(arr, target_shape):
    expansions = np.array([
        [0, target_shape[i] - arr.shape[i]]
        for i in range(len(target_shape))
    ])
    return np.pad(arr, expansions, mode='constant', constant_values=0)


def log_mel_fbe(fn, return_values=False, output_dir=None, sampling_rate=44100,
               nfft=1024, n_mels=64, duration=2, frame_length=1101, hop_length=661, window=None, force=False):

    f_to_mel = filters.mel(sr=sampling_rate, n_fft=nfft, n_mels=n_mels)
    overlap_length = frame_length - hop_length

    def extract(audio_fn):
        # Read and Resample the audio
        try:
            data, _ = librosa.core.load(audio_fn, sr=sampling_rate)
            data = normalize(data)
        except:
            return None

        # zero-padding to a multiple of <duration>-second chunks
        spectral_chunk_size = int(sampling_rate * duration)
        n_spectral_chunks = (1 + len(data) // spectral_chunk_size)
        target_length = sampling_rate * (duration * n_spectral_chunks)
        edge_pad_length = int(frame_length/2)
        data = np.pad(data, ((edge_pad_length, edge_pad_length)), mode='constant', constant_values=0)

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
        spectral_chunk_hop_length = int(int(sampling_rate * 0.500) / hop_length)
        n_frames_per_chunk = int(spectral_chunk_size / hop_length)
        spectral_chunks = [
            zero_fill(log_mel_spec[:, i:i + n_frames_per_chunk], (n_mels, n_frames_per_chunk, 1))
            for i in range(0, log_mel_spec.shape[1], spectral_chunk_hop_length)
        ]

        return spectral_chunks

    out_fns = list()
    if output_dir is not None:
        # create the output directory if it does not exist
        if not path.exists(output_dir):
            os.mkdir(output_dir)

        # extract and store the features at the output file(s), only if not already present
        fn_template = path.join(output_dir, path.split(fn)[-1] + ".npz")
        spectra = extract(fn)

        if spectra is None:
            return "BAD_" + fn_template

        def indexed_sub_fn(fn, i):
            prefix, ext = path.splitext(fn)
            return prefix + str(i) + ext

        for i, spec in enumerate(spectra):
            out_fns.append(indexed_sub_fn(fn_template, i))
            if not path.exists(out_fns[-1]) or force:
                np.savez_compressed(out_fns[-1], spec=spec)
    else:
        spectra = extract(fn)

    if return_values:
        if len(out_fns) > 0:
            return out_fns, spectra
        return spectra
    return out_fns
