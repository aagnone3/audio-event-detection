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
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import audio
from util import mp_with_pbar, load_config


def extract(method_name, file_list, **kwargs):
    fns = np.loadtxt(file_list, dtype='str')
    method = getattr(audio, method_name)
    mp_func = partial(method, **kwargs)
    feature_fns = mp_with_pbar(mp_func, fns, 16)

    return feature_fns


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(dest='input_fn', help='Path to file which lists wav paths to extract from.')
    parser.add_argument(dest='output_fn', help='Path to file which lists paths to extracted features.')
    parser.add_argument('-c', dest='config_fn', help='Path to configuration file.')
    return parser.parse_args()


def make_bad_output_fn(output_fn):
    base, ext = path.splitext(output_fn)
    return base + ".bad" + ext


if __name__ == '__main__':
    # parse arguments and perform extraction
    args = parse_args()
    params = load_config(args.config_fn)['data_loader']
    method_name = params.pop('extraction_method')
    fns = extract(method_name, args.input_fn, **params)

    # write good and bad output files
    n_bad = 0
    n_total = 0
    fn_bad_out = make_bad_output_fn(args.output_fn)
    with open(args.output_fn, 'w') as fp, open(fn_bad_out, 'w') as fp_bad:
        for fn in fns:
            n_total += 1
            if fn.startswith("BAD_"):
                n_bad += 1
                fp_bad.write(fn + '\n')
            else:
                fp.write(fn + '\n')

    # report status
    if n_bad == 0:
        os.remove(fn_bad_out)
        print('Done. No errors')
    else:
        print('Done. Failed to extract features for {}/{} files.'.format(n_bad, n_total))
