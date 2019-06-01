import os
import yaml
from functools import partial
import multiprocessing as mp
import numpy as np
np.random.seed(44)
from tqdm import tqdm


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


def load_config(fn):
    with open(fn, 'r') as fp:
        cfg = yaml.load(fp)
    return cfg
