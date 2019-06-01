import os
import sys
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
from sklearn.externals import joblib
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from util import load_config
from tqdm import tqdm


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(dest='input_fn', help='Path to file which lists hdf5 paths of features to use.')
    parser.add_argument(dest='output_fn', help='Path to file where scaler is persisted.')
    parser.add_argument('-c', dest='config_fn', help='Path to configuration file.')
    return parser.parse_args()


def make_feature_fn(fn, base_dir):
    return path.join(base_dir, path.basename(fn))


if __name__ == '__main__':
    args = parse_args()

    # get paths from file
    config = load_config(args.config_fn)
    with open(args.input_fn, 'r') as fp:
        feature_fns = [
            make_feature_fn(line.strip('\n'), config['features']['output_dir'])
            for line in fp.readlines()
        ]

    # train the scaler
    print("Training scaler")
    scaler = StandardScaler()
    for i, fn in tqdm(enumerate(feature_fns), total=len(feature_fns)):
        if path.exists(fn):
            with h5py.File(fn, 'r') as f:
                spec = np.hstack(f['data'][:, :, :, 0]).T
                scaler.partial_fit(spec)
        else:
            print("WARNING: {} does not exist".format(fn))

    joblib.dump(scaler, args.output_fn)
