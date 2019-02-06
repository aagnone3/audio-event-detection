import librosa
import multiprocessing as mp
from os import path
import pandas as pd
from functools import partial
from librosa import filters
from librosa.util import normalize
import scipy as sp
import numpy as np
np.random.seed(44)
from sklearn.preprocessing import StandardScaler
import joblib
from tqdm import tqdm

from feature_extraction.audio import log_mel_fbe


nfft = 1024
n_mels = 64
dim = (n_mels, 132, 1)
n_classes = 41
sampling_rate = 16000
duration = 2
input_length = int(sampling_rate * duration)
frame_size_ms = 25
hop_size_ms = 15
frame_length = int(sampling_rate * frame_size_ms / 1000.0)
hop_length = int(sampling_rate * hop_size_ms / 1000.0)
overlap_length = frame_length - hop_length
window = np.hanning(frame_length)


def extract(audio_dir, meta_fn, train_scaler=False):
    fns = list(df['fname'].map(lambda fn: path.join(audio_dir, fn)))
    cur_batch_size = len(fns)
    df = pd.read_csv(meta_fn)
    X = np.empty((cur_batch_size, *dim))
    y = np.empty((cur_batch_size, n_classes))
    f_to_mel = filters.mel(sr=sampling_rate, n_fft=nfft, n_mels=n_mels)

    with mp.Pool(mp.cpu_count()) as pool:
        mp_func = partial(log_mel_fbe, return_values=False, output_dir="features", sampling_rate=sampling_rate,
                         nfft=nfft, n_mels=n_mels, duration=duration,
                         frame_length=frame_length, hop_length=hop_length, window=window)
        feature_fns = pool.map_async(mp_func, fns).get()

    if train_scaler:
        scaler = StandardScaler()
        for i, fn in tqdm(enumerate(feature_fns), total=len(feature_fns)):
            spec = np.load(fn)['spec']
            scaler.partial_fit(spec[:,:,0].T)

        joblib.dump(scaler, "scaler.pkl")


if __name__ == '__main__':
    extract(
        "/home/aagnone/corpora/freesound-audio-tagging/audio_train",
        "/home/aagnone/corpora/freesound-audio-tagging/train_post_competition.csv",
        train_scaler=True
    )
    extract(
        "/home/aagnone/corpora/freesound-audio-tagging/audio_test",
        "/home/aagnone/corpora/freesound-audio-tagging/test_post_competition_scoring_clips.csv",
        train_scaler=False
    )
