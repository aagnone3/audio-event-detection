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


# load a pre-created mapping of label -> integer index
with open("etc/label_map.pkl", 'rb') as fp:
    label_map = pickle.load(fp)
    label_map['None'] = -1


def get_data(config, extract_real_time=False):
    # load data set meta data and labels
    data_dir = config["data_loader"]["data_dir"]
    train_file = path.join(data_dir, "train_post_competition.csv")
    train_data_dir = path.join(data_dir, "audio_train")
    test_data_dir = path.join(data_dir, "audio_test")
    test_file = path.join(data_dir, "test_post_competition.bad_removed.csv")

    all_train_data_df = pd.read_csv(train_file)
    all_train_data_df["label_idx"] = all_train_data_df.label.map(label_map.get).astype(int)

    train_idx = np.random.choice(all_train_data_df.index, size=int(0.7 * len(all_train_data_df)))
    train_mask = all_train_data_df.index.isin(train_idx)
    train_df, validation_df = all_train_data_df.loc[train_mask], all_train_data_df.loc[~train_mask]

    test_df = pd.read_csv(test_file)
    test_df["label_idx"] = test_df.label.map(label_map.get).astype(int)

    if len(train_df) == 0:
        raise ValueError("train_df is empty")
    if len(validation_df) == 0:
        raise ValueError("validation_df is empty")
    if len(test_df) == 0:
        raise ValueError("test_df is empty")

    if extract_real_time:
        train_df['fname'] = train_df['fname'].map(lambda fn: path.join(train_data_dir, fn))
        validation_df['fname'] = validation_df['fname'].map(lambda fn: path.join(train_data_dir, fn))
        test_df['fname'] = test_df['fname'].map(lambda fn: path.join(test_data_dir, fn))
    else:
        data_dir = "features"
        train_df['fname'] = train_df['fname'].map(lambda fn: path.join(data_dir, fn) + '.npz')
        validation_df['fname'] = validation_df['fname'].map(lambda fn: path.join(data_dir, fn) + '.npz')
        test_df['fname'] = test_df['fname'].map(lambda fn: path.join(data_dir, fn) + '.npz')

    return train_df, validation_df, test_df


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

    # get data sets
    train_df, validation_df, test_df = get_data(config, extract_real_time=False)

    # create data generators for the data sets
    loader_params = config["data_loader"]
    training_generator = data_loader_cls(train_df['fname'].values, train_df['label_idx'].values, shuffle=True,
                                         **loader_params)
    validation_generator = data_loader_cls(validation_df['fname'].values, validation_df['label_idx'].values,
                                           **loader_params)
    testing_generator = data_loader_cls(test_df['fname'].values, test_df['label_idx'].values,
                                        **loader_params)

    # train ze model
    model = model_cls(training_generator.feature_dim, training_generator.n_classes)
    trainer = trainer_cls(
        config["exp"]["name"],
        model.model,
        config["callbacks"],
        **config["trainer"]
    )
    trainer.train(
        training_generator,
        validation_generator
    )
    show_metrics(trainer.model, trainer.test(testing_generator))


if __name__ == '__main__':
    main()
