import json

import numpy as np
import statistics as sts

import TP5_v2.src.utils.parser as parser
import TP5_v2.src.utils.functions as functions
import TP5_v2.src.utils.tools as tools
import TP5_v2.src.models.autoencoder as ae

with open("TP5_v2/src/config.json") as file:
    config = json.load(file)

# static non changeable vars
error_threshold: float = config["error_threshold"]

# read the files and get the dataset. There is no need to normalize data at this exercise
full_dataset, _ = parser.read_file(config["file"], config["system_threshold"])

# activation function and its derived
act_funcs = functions.get_activation_functions(config["system"], config["beta"])

# normalize data
if config["normalize"]:
    full_dataset = parser.normalize_data(full_dataset)

# extract the last % of the dataset
dataset, rest = parser.extract_subset(full_dataset, config["training_ratio"])

# initializes the auto-encoder
auto_encoder = ae.AutoEncoder(*act_funcs, config["mid_layout"], len(dataset[0]), config["latent_dim"],
                              config["momentum"], config["alpha"])

plot_bool = bool(config["plot"])

# initialize plotter
if plot_bool:
    tools.init_plotter()

# get pm from config
pm: float = config["denoising"]["pm"]

# vars for plotting
ep_list = []
err_list = []

# train auto-encoder
for ep in range(config["epochs"]):

    # randomize the dataset everytime
    dataset = parser.randomize_data(dataset, config["data_random_seed"])

    # train for this epoch
    for data in dataset:
        auto_encoder.train(parser.add_noise(data, pm), data, config["eta"])

    # apply the changes
    auto_encoder.update_weights()

    # calculate error
    error: float = auto_encoder.compute_error(parser.add_noise_dataset(dataset, pm), dataset)
    if error < config["error_threshold"]:
        break

    if ep % 50 == 0:
        print(f'Iteration {ep}, error {error}')

    # add error to list
    ep_list.append(ep)
    err_list.append(error)

# plot error vs epoch
if plot_bool:
    tools.plot_values(ep_list, 'epoch', err_list, 'error', sci_y=False)

# labels for printing (use with full_dataset)
labels: [] = ['@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '^', '_']

PM_ITER = 50

pm_values = [pm / 4, pm, pm * 2.5]
x_superlist = []
err_superlist = []
leg_list = ['pm=0,0625', 'pm=0,25', 'pm=0,625']
for pm_it in pm_values:
    err_mean: [] = []
    for data in full_dataset:
        aux: [] = []
        for i in range(PM_ITER):
            noisy_res = auto_encoder.forward_pass(parser.add_noise(data, pm_it))
            aux.append(np.sum(abs(np.around(noisy_res[1:]) - data[1:])) / len(data[1:]))
        letter_err_mean = sts.mean(aux)
        err_mean.append(letter_err_mean)
    x_superlist.append(range(len(full_dataset)))
    err_superlist.append(err_mean)
    print(f'Using pm={pm_it}, error mean is {sts.mean(err_mean)}')

if plot_bool:
    tools.plot_multiple_values(x_superlist, 'Letter', err_superlist, 'Invalid bits', leg_list, sci_y=False, xticks=labels, min_val_y=0, max_val_y=1)
    tools.plot_stackbars(x_superlist, 'Letter', err_superlist, 'Invalid bits', leg_list, sci_y=False, xticks=labels, min_val_y=0, max_val_y=1)

    # hold execution
    tools.hold_execution()
