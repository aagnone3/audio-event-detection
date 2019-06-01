import os
import h5py
import librosa
from os import path
from functools import partial
from librosa import filters
from librosa.util import normalize
import multiprocessing as mp
import scipy as sp
import numpy as np
np.random.seed(44)
from sklearn.preprocessing import StandardScaler
import joblib
from tqdm import tqdm

from audio import *


nfft = 2048
n_freq_bins = 64
# sampling_rate = 44100
sampling_rate = 33000
frame_length_s = 0.040
hop_length_s = 0.020
chunk_length_s = 1.0
chunk_hop_length_s = 1.0
duration_s = 10.0


def mp_with_pbar(func, args, n_processes = 2):
    res_list = []
    if os.environ.get("DISABLE_MP", '') == '1':
        with tqdm(total = len(args)) as pbar:
            for i, res in tqdm(enumerate(map(func, args))):
                pbar.update()
                res_list.append(res)
    else:
        p = mp.Pool(n_processes)
        with tqdm(total = len(args)) as pbar:
            for i, res in tqdm(enumerate(p.imap_unordered(func, args))):
                pbar.update()
                res_list.append(res)

            pbar.close()
            p.close()
            p.join()
    return res_list


def extract(file_list, train_scaler=False):
    fns = np.loadtxt(file_list, dtype='str')
    cur_batch_size = len(fns)
    f_to_mel = filters.mel(sr=sampling_rate, n_fft=nfft, n_freq_bins=n_freq_bins)

    print("Extracting features")
    mp_func = partial(log_filterbank_energy_sequence, output_dir="features", sampling_rate=sampling_rate,
                      nfft=nfft, n_freq_bins=n_freq_bins, frame_length_s=frame_length_s,
                      spectral_frame_length_s=chunk_length_s, spectral_hop_length_s=chunk_hop_length_s,
                      hop_length_s=hop_length_s, force=True, mel_scale=True)
    feature_fns = mp_with_pbar(mp_func, fns, 16)

    if train_scaler:
        print("Training scaler")
        scaler = StandardScaler()
        for i, fn in tqdm(enumerate(feature_fns), total=len(feature_fns)):
            with h5py.File(fn, 'r') as f:
                spec = f['data']
            scaler.partial_fit(spec[:,:,0].T)

        joblib.dump(scaler, "scaler.pkl")

    return feature_fns


if __name__ == '__main__':
    values = [
        ['data/all_wavs.lst', 'bad_features.lst', False],
    ]
    for fn_in, fn_out, train_scaler in values:
        fns = extract(
            fn_in,
            train_scaler=train_scaler
        )
        base, ext = path.splitext(fn_out)
        fn_bad_out = base + ".bad" + ext
        n_bad = 0
        n_total = 0
        with open(fn_out, 'w') as fp, open(fn_bad_out, 'w') as fp_bad:
            for fn in fns:
                n_total += 1
                if fn.startswith("BAD_"):
                    n_bad += 1
                    fp_bad.write(fn + '\n')
                else:
                    fp.write(fn + '\n')

        if n_bad == 0:
            os.remove(fn_bad_out)
            print('Done. No errors')
        else:
            print('Done. Failed to extract features for {}/{} files.'.format(n_bad, n_total))
