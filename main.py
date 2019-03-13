import os
import yaml
import argparse
import pickle
from os import path
import pandas as pd
import numpy as np

from utils import factory, create_dirs
np.random.seed(44)


def process_config(fn):
    with open(fn, 'r') as fp:
        config = yaml.load(fp)

    config["callbacks"]["tensorboard_log_dir"] = os.path.join(
        "experiments",
        config["exp"]["name"],
        "logs/"
    )
    config["callbacks"]["checkpoint_dir"] = os.path.join(
        "experiments",
        config["exp"]["name"],
        "checkpoints/"
    )
    return config


def get_data(config, extract_real_time=False):
    # define directory names
    train_data_dir = config["data_loader"]["train"]["data_dir"]
    eval_data_dir = config["data_loader"]["eval"]["data_dir"]

    # load data set meta data and labels
    all_train_data_df = pd.read_csv(config["data_loader"]["train"]["meta_file"]).fillna('nan')

    train_idx = np.random.choice(all_train_data_df.index, size=int(0.7 * len(all_train_data_df)))
    train_mask = all_train_data_df.index.isin(train_idx)
    train_df, training_val_df = all_train_data_df.loc[train_mask], all_train_data_df.loc[~train_mask]

    eval_df = pd.read_csv(config["data_loader"]["eval"]["meta_file"]).fillna('nan')

    if len(train_df) == 0:
        raise ValueError("train_df is empty")
    if len(training_val_df) == 0:
        raise ValueError("training_val_df is empty")
    if len(eval_df) == 0:
        raise ValueError("eval_df is empty")

    if extract_real_time:
        train_df['filename'] = train_df['filename'].map(lambda fn: path.join(train_data_dir, fn))
        training_val_df['filename'] = training_val_df['filename'].map(lambda fn: path.join(train_data_dir, fn))
        eval_df['fname'] = eval_df['filename'].map(lambda fn: path.join(eval_data_dir, fn))
    else:
        data_dir = "features"
        train_df['filename'] = train_df['filename'].map(lambda fn: path.join(data_dir, fn) + '.hdf5')
        training_val_df['filename'] = training_val_df['filename'].map(lambda fn: path.join(data_dir, fn) + '.hdf5')
        eval_df['filename'] = eval_df['filename'].map(lambda fn: path.join(data_dir, fn) + '.hdf5')

    return train_df, training_val_df, eval_df


def show_metrics(model, evaluation):
    for metric_name, metric_value in zip(model.metrics_names, evaluation):
        print("{}: {}".format(metric_name, metric_value))


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        dest='config',
        metavar='C',
        default='None',
        help='The Configuration file')
    args = argparser.parse_args()
    return args


def main():
    args = get_args()
    config = process_config(args.config)

    # create the experiments dirs
    create_dirs([config["callbacks"]["tensorboard_log_dir"], config["callbacks"]["checkpoint_dir"]])

    # resolve classes
    data_loader_cls = factory("data_loaders.{}".format(config["data_loader"]["name"]))
    model_cls = factory("models.{}".format(config["model"]["name"]))
    trainer_cls = factory("trainers.{}".format(config["trainer"]["name"]))

    # create data generators for the data sets
    loader_params = config["data_loader"]
    training_generator = data_loader_cls("train", shuffle=True, **loader_params)
    eval_generator = data_loader_cls("eval", **loader_params)

    # train ze model
    model_params = config['model']
    print(training_generator.feature_dim)
    model = model_cls(training_generator.feature_dim, training_generator.n_classes, **model_params)
    trainer = trainer_cls(
        config["exp"]["name"],
        model.model,
        config["callbacks"],
        **config["trainer"]
    )
    trainer.train(
        training_generator,
        eval_generator
    )
    # show_metrics(trainer.model, trainer.test(eval_generator))


if __name__ == '__main__':
    main()
