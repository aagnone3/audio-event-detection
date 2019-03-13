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

from audio import log_mel_fbe


nfft = 1024
n_mels = 64
dim = (n_mels, 132, 1)
sampling_rate = 16000
spectral_frame_length_s = 0.960
spectral_hop_length_s = 0.250
frame_length_s = 0.025
hop_length_s = 0.010


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
    f_to_mel = filters.mel(sr=sampling_rate, n_fft=nfft, n_mels=n_mels)

    print("Extracting features")
    mp_func = partial(log_mel_fbe, output_dir="features", sampling_rate=sampling_rate,
                      nfft=nfft, n_mels=n_mels, spectral_frame_length_s=spectral_frame_length_s,
                      spectral_hop_length_s=spectral_hop_length_s, frame_length_s=frame_length_s,
                      hop_length_s=hop_length_s, force=False)
    feature_fns = mp_with_pbar(mp_func, fns, mp.cpu_count())

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
        ['asc.lst', 'asc.features.lst', False],
        # ['train.synth.lst', 'train.synth.features.lst', False],
        # ['eval.synth.lst', 'eval.synth.features.lst', False],
    ]
    for fn_in, fn_out, train_scaler in values:
        fns = extract(
            fn_in,
            train_scaler=train_scaler
        )
        base, ext = path.splitext(fn_out)
        fn_bad_out = base + ".bad" + ext
        has_bad = False
        with open(fn_out, 'w') as fp, open(fn_bad_out, 'w') as fp_bad:
            for fn in fns:
                if fn.startswith("BAD_"):
                    if not has_bad:
                        has_bad = True
                    fp_bad.write(fn + '\n')
                else:
                    fp.write(fn + '\n')

        if not has_bad:
            os.remove(fn_bad_out)
