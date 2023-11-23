import json
import numpy as np
import TP5_v2.src.utils.parser as parser
import TP5_v2.src.utils.functions as functions
import TP5_v2.src.utils.tools as tools
import TP5_v2.src.models.autoencoder as ae

import matplotlib.pyplot as plt

LETTER_WIDTH = 5

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

# randomize w if asked
if bool(config["randomize_w"]):
    auto_encoder.initialize_weights(config["randomize_w_ref"], config["randomize_w_by_len"])

plot_bool = bool(config["plot"])

# initialize plotter
if plot_bool:
    tools.init_plotter()
    plt.ion()
    plt.show()

# use minimizer if asked
if config["optimizer"] != "None" and config["optimizer"] != "":
    # randomize the dataset
    dataset = parser.randomize_data(dataset, config["data_random_seed"])
    # train with minimize
    auto_encoder.train_with_minimization(dataset, dataset, config["trust"], config["use_trust"], config["optimizer"],
                                 config["optimizer_iter"], config["optimizer_fev"])
    # plot error vs opt step
    tools.plot_values(range(len(auto_encoder.optimization_errors)), 'opt step', auto_encoder.optimization_errors, 'error', sci_y=False)
else:
    # vars for plotting
    ep_list = []
    err_list = []

    # train auto-encoder
    for ep in range(config["epochs"]):

        # randomize the dataset everytime
        dataset = parser.randomize_data(dataset, config["data_random_seed"])

        # train for this epoch
        for data in dataset:
            auto_encoder.train(data, data, config["eta"])

        # apply the changes
        auto_encoder.update_weights()

        # calculate error
        err = auto_encoder.compute_error(dataset, dataset, config["trust"], config["use_trust"])

        if err < config["error_threshold"]:
            break

        if ep % 50 == 0:
            print(f'Iteration {ep}, error {err}')

        # add error to list
        ep_list.append(ep)
        err_list.append(err)

    # plot error vs epoch
    if plot_bool:
        tools.plot_values(ep_list, 'epoch', err_list, 'error', sci_y=False)

# labels for printing (use with full_dataset)
labels: [] = ['@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
              'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '^', '_']

# show latent space given the input
aux: [] = []
for data in full_dataset:
    aux.append(auto_encoder.encode(data))
latent_space: np.ndarray = np.array(aux)
# plot latent space
tools.plot_latent_space(latent_space, labels, -1, 1)

# hold execution
if plot_bool:
    # utils.hold_execution()
    plt.pause(0.001)

while True:
    print('--------------------------------------------')
    for i, l in enumerate(labels):
        print(f'{i}: {l}', end='\t')
        if i % 10 == 9: print('')

    index_1 = int(input("\nIngrese un indice de letra: "))
    index_2 = int(input("\nIngrese otro indice de letra: "))

    # generate a new letter not from the dataset. Creates a new Z between the first two
    new_latent_space: np.ndarray = np.sum([latent_space[index_1], latent_space[index_2]], axis=0) / 2
    new_letter: np.ndarray = auto_encoder.decode(new_latent_space)

    tools.print_pattern(full_dataset[index_1, 1:], LETTER_WIDTH)
    print('\n--------------------------------------------')
    tools.print_pattern(full_dataset[index_2, 1:], LETTER_WIDTH)
    print('\n--------------------------------------------')
    print(new_letter)
    print('\n--------------------------------------------')
    tools.print_pattern(np.around(new_letter[1:]), LETTER_WIDTH)
