import math
import random

import numpy as np


def read_file(data_name: str,  threshold: int = 1) -> (np.ndarray, object):
    number_class = int
    try:
        # read, save training data
        data = np.array(parse_file_set(data_name, number_class, threshold, training=True))
    except ValueError:
        number_class = float
        data = np.array(parse_file_set(data_name, number_class, threshold, training=True))

    return data, number_class


def parse_file_set(file_name: str, number_class=float, threshold: int = 1, training: bool = False) -> []:
    training_file = open(file_name, "r")
    file_data = []

    # each line could have several numbers
    # initialize or not with threshold
    for line in training_file:
        line_data = [number_class(threshold)] if training else []
        # for each number, append it
        for n in line.split():
            line_data.append(number_class(n))
        file_data.append(line_data)
    return file_data


def normalize_data(data: np.ndarray) -> np.ndarray:
    return (2. * (data - np.min(data)) / np.ptp(data) - 1)


def randomize_data(data: np.ndarray, seed: int) -> np.ndarray:
    aux: np.ndarray = np.c_[data.reshape(len(data), -1)]
    if seed != 0:
        np.random.seed(seed)
    np.random.shuffle(aux)
    return aux[:, :data.size // len(data)].reshape(data.shape)


def extract_subset(data: np.ndarray, ratio: int) -> (np.ndarray, np.ndarray):
    dataset: np.ndarray = data
    rest_len: int = int(len(data) * (1 - ratio))
    rest_data = []

    for i in range(rest_len):
        # move from full list to test
        rest_data.append(data[i])

    # remove data from test
    return np.delete(dataset, np.arange(0, len(rest_data)), 0), np.array(rest_data)


def add_noise(data: np.ndarray, prob: float) -> np.ndarray:
    resp: np.ndarray = data
    # skip bias
    for i in range(1, len(data)):
        if np.random.uniform() < prob:
            resp[i] = 1 - data[i]
    return resp


def add_noise_dataset(dataset: np.ndarray, prob: float) -> np.ndarray:
    ret: [] = []
    for data in dataset:
        ret.append(add_noise(data, prob))

    return np.asarray(ret)
