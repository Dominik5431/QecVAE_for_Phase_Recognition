import sys
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import sem
from parameters import parameters
import torch
from torch.optim import Optimizer
from tqdm import tqdm

import src.NN.utils.functions as functions
from src.NN.dataset import BitFlipSurfaceData
from src.NN.net import VariationalAutoencoder
from src.NN.utils.plotter import scatter_latent_var
from src.NN.train import train
from src.NN.test import test_model_latent_space, test_model_reconstruction_error
from src.NN.utils import plotter
from src.NN.predictions import Predictions

import numpy as np
import gc
import argparse
import logging


def predicting(noise_min, noise_max, resolution, n_pred, nn):
    noise_arr = np.arange(noise_min, noise_max, resolution)
    predics = np.zeros((len(noise_arr), 2))
    predics_err = np.zeros((len(noise_arr), 2))
    for k, noise in enumerate(noise_arr):
        syndromes = (Dataset.DepolarizingSurfaceData(parsed.dist, [noise], "pred_data")
                     .generate_data(n_pred, 1)
                     .prepare_data(LAYOUT)
                     .get_syndromes(parsed.batch_size))
        temp = nn.predict(syndromes)  # temp has shape (len(syndromes),2)
        del syndromes
        gc.collect()
        predics[k, :] = np.mean(temp, axis=0)
        predics_err[k, :] = sem(temp, axis=0)
    return predics, predics_err


def make_optimizer(lr):
    return lambda params: torch.optim.Adam(params, lr=lr)


name_dict = "latents_bitflip_test_fc_only"


def loss_func(output, target: torch.Tensor) -> torch.Tensor:
    # reproduction_loss = torch.nn.functional.binary_cross_entropy(output, target, reduction='sum') -> try if this works better
    # kldivloss = torch.nn.functional.kl_div(output, target, reduction='sum')  # is this really doing the right thing? -> no defined as pointwise KL divergence loss, put in tensors of points that are distributed according to the distributions Q and P whose KL divergence loss is to be calculated
    reproduction_loss = torch.nn.MSELoss()
    recon, mean, logvar = output
    kldivloss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return reproduction_loss(recon, target) + kldivloss


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    plt.set_loglevel("notset")

    LR, NOISE_MODEL, NUM_EPOCHS, BATCH_SIZE, DATA_SIZE, DISTANCE, LOAD_DATA, SAVE_DATA, NOISES_TRAINING, NOISES_TESTING, RATIO, LATENT_DIMS = parameters()
    task = 3
    if task == 0:  # Create data
        logging.debug("Create data.")
        if NOISE_MODEL == 'BitFlip':
            data = BitFlipSurfaceData(distance=DISTANCE, noises=NOISES_TRAINING, name="BFS_1-{0}".format(DISTANCE),
                                      num=DATA_SIZE, load=LOAD_DATA)
            data.save()
    elif task == 1:  # Training a network
        logging.debug("Get data.")
        if NOISE_MODEL == 'BitFlip':
            data_train, data_val = BitFlipSurfaceData(distance=DISTANCE, noises=NOISES_TRAINING,
                                                      name="BFS_Test-{0}".format(DISTANCE),
                                                      num=DATA_SIZE, load=LOAD_DATA).get_train_test_data(RATIO)
        logging.debug("Train NN.")
        model = VariationalAutoencoder(LATENT_DIMS, DISTANCE, "VAE-fc_only-{0}".format(DISTANCE))
        model = train(model, make_optimizer(LR), loss_func, NUM_EPOCHS, BATCH_SIZE, data_train, data_val)
        model.save()

    elif task == 2:  # Testing a network
        logging.debug("Get data.")
        '''if NOISE_MODEL == 'BitFlip':
            data_train, data_val = BitFlipSurfaceData(distance=DISTANCE, noises=NOISES,
                                                      name="BFS_Test-{0}".format(DISTANCE),
                                                      num=DATA_SIZE, load=LOAD_DATA).get_train_test_data(RATIO)'''
        logging.debug("Evaluate latent space.")
        model = VariationalAutoencoder(LATENT_DIMS, DISTANCE, "VAE-fc_only-{0}".format(DISTANCE))
        model.load()
        # Use dictionary with noise value and return values to store return data from VAE while testing
        latents = Predictions(name=name_dict)
        latents.load()
        results = {}
        for noise in tqdm(NOISES_TESTING):
            data_test = BitFlipSurfaceData(distance=DISTANCE, noises=[noise],
                                           name="BFS_Testing-{0}".format(DISTANCE),
                                           num=20, load=False)
            results[noise] = test_model_latent_space(model, data_test)  # z_mean, z_logvar, z_bar, z_bar_var
        latents.add(DISTANCE, results)
        latents.save()

    elif task == 3:  # plot latent space, computed in task 2
        scatter_latent_var(Predictions(name=name_dict).load().get_dict())

    elif task == 10:  # Obtain T_C, plot T_C vs d ,TODO
        #noises = np.arange(0.01, 0.30, 0.002)
        noises = np.arange(0.01, 0.30, 0.01)
        predictions = Predictions.Predictions().load().get_dict()
        # print(np.shape(predictions))
        # Plot.Plotter.plot_prediction(predictions, noises)

        res = functions.get_pcs(predictions, noises)

        plt.plot(res[:, 0], res[:, 1], c='blue', marker='x', linewidth=1)
        # plt.errorbar(res[:, 0], res[:, 1], res[:, 2], c='blue', marker='x', linewidth=1)
        print(res[:, 2])
        popt, pcov = curve_fit(f=functions.linear, xdata=res[:, 0], ydata=res[:, 1])  # , sigma = res[:,2],
        # absolute_sigma=True)
        plt.plot(np.arange(0, 0.101, 0.001), functions.linear(np.arange(0, 0.101, 0.001), popt[0], popt[1]))
        plt.xlabel('1/L')
        plt.ylabel('p_c')
        print("pcritical=", popt[1], "+-", pcov[1, 1])
        plt.show()

    elif task == 4:  # Perform data collapse
        logging.debug("Start data collapse.")
        predictions = Predictions.Predictions().load().get_dict()
        # Plot.Plotter.plot_prediction(predictions, noises)
        res = functions.data_collapse(NOISES, predictions)
        pc = res.x[0]
        nu = res.x[1]
        print(res.x)
        Plot.Plotter.plot_collapsed(predictions, NOISES, pc, nu)
        print(res.x)

    elif task == 5:  # Plot graphs
        predictions = Predictions.Predictions().load().get_dict()
        plotter.plot_prediction(predictions, NOISES)

    elif task == 6:  # Show network structure
        if parsed.noise == 'Bit-Flip':
            plot_model(NN.CNNBitFlip(parsed.dist, FILTERS, LAYOUT).model, to_file="CNN_BFS.png", show_shapes=True)
        elif parsed.noise == 'Depolarizing':
            plot_model(NN.CNNDepolarizing(parsed.dist, FILTERS, LAYOUT).model, to_file="CNN_D.png", show_shapes=True)

    elif task == 7:  # Plot history of a training process
        path = 'files/prediction_depolarizing_39.pkl'
        #prediction = Predictions.Predictions(path)
        prediction.load()
        dict = prediction.get_dict()
        out = dict.get('39')
        noise1 = np.arange(0.01, 0.30, 0.002)
        print(noise1[::5])
        print(np.arange(0.01, 0.30, 0.01))
        out_new = out[noise1[::5]]
        path = 'files/prediction_depolarizing_0.pkl'
        #prediction = Predictions.Predictions(path)
        prediction.load()
        prediction.add(39, out_new)
        prediction.save()

    elif task == 8:  # Plot QEC code
        if NOISE_MODEL == 'Bit-Flip':
            code = ErrorCode.BitFlipSurface(parsed.dist, 0.1)
            code.circuit_to_png()
        elif parsed.noise == 'Depolarizing':
            code = ErrorCode.DepolarizingSurface(parsed.dist, 0.1)
            code.circuit_to_png()

    else:
        print("Unknown task number.")
        exit(-1)
