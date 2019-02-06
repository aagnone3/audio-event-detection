import os
from os import path
import numpy as np
import scipy as sp
import multiprocessing as mp
from functools import partial

import librosa
from librosa.util import normalize
from librosa import filters



def log_mel_fbe(fn, return_values=False, output_dir=None, sampling_rate=44100,
               nfft=1024, n_mels=64, duration=2, frame_length=1101, hop_length=661, window=None):

    f_to_mel = filters.mel(sr=sampling_rate, n_fft=nfft, n_mels=n_mels)
    overlap_length = frame_length - hop_length
    input_length = int(sampling_rate * duration)

    def extract(audio_fn):
        # Read and Resample the audio
        try:
            data, _ = librosa.core.load(audio_fn, sr=sampling_rate)
            data = normalize(data)
        except:
            return None

        # Random offset / Padding
        # TODO get all windows
        if len(data) > input_length:
            max_offset = len(data) - input_length
            offset = np.random.randint(max_offset)
            data = data[offset:(input_length+offset)]
        else:
            if input_length > len(data):
                max_offset = input_length - len(data)
                offset = np.random.randint(max_offset)
                data = np.pad(data, (offset, input_length - len(data) - offset), "constant")

        f, t, Sxx = sp.signal.spectrogram(
            data,
            fs=sampling_rate,
            window=window,
            nperseg=frame_length,
            noverlap=overlap_length,
            nfft=nfft
        )
        mel_spec = f_to_mel.dot(Sxx)
        log_mel_spec = np.log(1e-8 + mel_spec)
        log_mel_spec = np.expand_dims(log_mel_spec, axis=-1)
        return log_mel_spec

    out_fn = None
    if output_dir:
        # create the output directory if it does not exist
        if not path.exists(output_dir):
            os.mkdir(output_dir)

        # store the features at the output file, if not already present
        out_fn = path.join(output_dir, path.split(fn)[-1] + ".npz")
        log_mel_spec = extract(fn)
        if log_mel_spec is None:
            return "BAD_" + out_fn

        if not path.exists(out_fn):
            np.savez_compressed(out_fn, spec=extract(fn))
    else:
        log_mel_spec = extract(fn)

    if return_values:
        if out_fn:
            return out_fn, log_mel_spec
        return log_mel_spec
    elif out_fn:
        return out_fn
