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

from audio import log_mel_fbe


nfft = 1024
n_mels = 64
dim = (n_mels, 132, 1)
n_classes = 41
sampling_rate = 16000
duration = 3
input_length = int(sampling_rate * duration)
frame_size_ms = 25
hop_size_ms = 10
frame_length = int(sampling_rate * frame_size_ms / 1000.0)
hop_length = int(sampling_rate * hop_size_ms / 1000.0)
overlap_length = frame_length - hop_length
window = np.hanning(frame_length)


def mp_with_pbar(func, args, n_processes = 2):
    p = mp.Pool(n_processes)
    res_list = []
    with tqdm(total = len(args)) as pbar:
        for i, res in tqdm(enumerate(p.imap_unordered(func, args))):
            pbar.update()
            res_list.append(res)

        pbar.close()
        p.close()
        p.join()
    return res_list


def extract(audio_dir, meta_fn, train_scaler=False):
    df = pd.read_csv(meta_fn)
    fns = list(df['fname'].map(lambda fn: path.join(audio_dir, fn)))
    cur_batch_size = len(fns)
    df = pd.read_csv(meta_fn)
    X = np.empty((cur_batch_size, *dim))
    y = np.empty((cur_batch_size, n_classes))
    f_to_mel = filters.mel(sr=sampling_rate, n_fft=nfft, n_mels=n_mels)

    print("Extracting features")
    mp_func = partial(log_mel_fbe, return_values=False, output_dir="features", sampling_rate=sampling_rate,
                     nfft=nfft, n_mels=n_mels, duration=duration,
                     frame_length=frame_length, hop_length=hop_length, window=window, force=True)
    feature_fn_groups = mp_with_pbar(mp_func, fns, mp.cpu_count())

    feature_fns = list()
    for group in feature_fn_groups:
        feature_fns += group

    if train_scaler:
        print("Training scaler")
        scaler = StandardScaler()
        for i, fn in tqdm(enumerate(feature_fns), total=len(feature_fns)):
            spec = np.load(fn)['spec']
            scaler.partial_fit(spec[:,:,0].T)

        joblib.dump(scaler, "scaler.pkl")


if __name__ == '__main__':
    extract(
        "/corpora/freesound-audio-tagging/audio_train",
        "/corpora/freesound-audio-tagging/train_post_competition.csv",
        train_scaler=True
    )
    extract(
        "/corpora/freesound-audio-tagging/audio_test",
        "/corpora/freesound-audio-tagging/test_post_competition_scoring_clips.csv",
        train_scaler=False
    )
