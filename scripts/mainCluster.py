import sys

import numpy as np
import torch
from torch import nn

from src.nn import VisionTransformer
from src.nn.data import BitFlipToricData, DepolarizingToricData
from src.nn.net import VariationalAutoencoder
from src.nn.data.predictions import Predictions
from src.nn.net.cnn import CNN
from src.nn.test import test_model_latent_space, test_model_reconstruction_error, test_model_predictions
from src.nn.train import train, train_supervised
from src.nn.utils.loss import loss_func
from src.nn.utils.optimizer import make_optimizer
from parameters import parameters
from tqdm import tqdm

STRUCTURES = ['standard', 'simple', 'skip', 'ising']
DISTANCES = [15, 21, 27, 33, 37, 43]


def train_model():
    data_train = None
    data_val = None
    if NOISE_MODEL == 'BitFlip':
        # data_train, data_val = BitFlipRotatedSurfaceData(distance=DISTANCE, noises=NOISES_TRAINING,
        data_train, data_val = (BitFlipToricData(distance=DISTANCE, noises=NOISES_TRAINING,
                                                 name=name_data.format(DISTANCE),
                                                 load=LOAD_DATA,
                                                 random_flip=random_flip,
                                                 sequential=sequential, cluster=CLUSTER,
                                                 supervised=SUPERVISED)
                                .training()
                                .initialize(num=DATA_SIZE)
                                .get_train_test_data(RATIO))
    elif NOISE_MODEL == 'Depolarizing':
        data_train, data_val = (DepolarizingToricData(distance=DISTANCE, noises=NOISES_TRAINING,
                                                      name=name_data.format(DISTANCE),
                                                      load=LOAD_DATA,
                                                      random_flip=random_flip,
                                                      sequential=sequential, cluster=CLUSTER)
                                .training()
                                .initialize(num=DATA_SIZE)
                                .get_train_test_data(RATIO))
    assert data_train is not None
    assert data_val is not None
    if STRUCTURE == 'cnn':
        model = CNN(distance=DISTANCE, channels=1 if NOISE_MODEL == 'BitFlip' else 2, name=name_NN, cluster=CLUSTER)
    elif STRUCTURE == 'vision-transformer':
        model = VisionTransformer(cluster=CLUSTER)
    else:
        model = VariationalAutoencoder(LATENT_DIMS, DISTANCE, name_NN.format(DISTANCE), structure=STRUCTURE,
                                       noise=NOISE_MODEL, cluster=CLUSTER)
    if not SUPERVISED:
        model = train(model, make_optimizer(LR), loss_func, NUM_EPOCHS, BATCH_SIZE, data_train, data_val)
    else:
        model = train_supervised(model, make_optimizer(LR), nn.BCELoss, NUM_EPOCHS, BATCH_SIZE,
                                 data_train, data_val)


def latents():
    if STRUCTURE == 'cnn':
        model = CNN(distance=DISTANCE, channels=1 if NOISE_MODEL == 'BitFlip' else 2, name=name_NN, cluster=CLUSTER)
    elif STRUCTURE == 'vision-transformer':
        model = VisionTransformer(cluster=CLUSTER)
    else:
        model = VariationalAutoencoder(LATENT_DIMS, DISTANCE, name_NN.format(DISTANCE), structure=STRUCTURE,
                                       noise=NOISE_MODEL, cluster=CLUSTER)
    model.load()
    # Use dictionary with noise value and return values to store return data from VAE while testing
    if SUPERVISED:
        latents = Predictions(name_dict_predictions, cluster=CLUSTER)
    else:
        latents = Predictions(name=name_dict_latent, cluster=CLUSTER)
    latents.load()
    results = {}
    for noise in tqdm(NOISES_TESTING):
        data_test = None
        if NOISE_MODEL == 'BitFlip':
            # data_test = BitFlipRotatedSurfaceData(distance=DISTANCE, noises=[noise],
            data_test = (BitFlipToricData(distance=DISTANCE, noises=[noise],
                                          name="BFS_Testing-{0}".format(DISTANCE),
                                          load=False, random_flip=random_flip, sequential=sequential,
                                          cluster=CLUSTER, supervised=SUPERVISED)
                         .eval()
                         .initialize(num=1000))
        elif NOISE_MODEL == 'Depolarizing':
            data_test = (DepolarizingToricData(distance=DISTANCE, noises=[noise],
                                               name="DS_Testing-{0}".format(DISTANCE),
                                               load=False, random_flip=random_flip, sequential=sequential,
                                               cluster=CLUSTER)
                         .eval()
                         .initialize(num=1000))
        assert data_test is not None
        if not SUPERVISED:
            results[noise] = test_model_latent_space(model, data_test)  # z_mean, z_log_var, z, flips
        else:
            results[noise] = test_model_predictions(model, data_test)
    latents.add(DISTANCE, results)
    latents.save()


def reconstructions():
    model = VariationalAutoencoder(LATENT_DIMS, DISTANCE, name_NN.format(DISTANCE), structure=STRUCTURE,
                                   noise=NOISE_MODEL, cluster=CLUSTER)
    model.load()
    # Use dictionary with noise value and return values to store return data from VAE while testing
    reconstructions = Predictions(name=name_dict_recon, cluster=CLUSTER)
    reconstructions.load()
    results = {}
    for noise in tqdm(NOISES_TESTING):
        data_test = None
        if NOISE_MODEL == 'BitFlip':
            # data_test = BitFlipRotatedSurfaceData(distance=DISTANCE, noises=[noise],
            data_test = (BitFlipToricData(distance=DISTANCE, noises=[noise],
                                          name="BFS_Testing-{0}".format(DISTANCE),
                                          load=False, random_flip=random_flip, sequential=sequential, cluster=CLUSTER)
                         .eval()
                         .initialize(num=1000))
        elif NOISE_MODEL == 'Depolarizing':
            data_test = (DepolarizingToricData(distance=DISTANCE, noises=[noise],
                                               name="DS_Testing-{0}".format(DISTANCE),
                                               load=False, random_flip=random_flip, sequential=sequential,
                                               cluster=CLUSTER)
                         .eval()
                         .initialize(num=1000))
        results[noise] = test_model_reconstruction_error(model, data_test,
                                                         torch.nn.MSELoss(reduction='none'))  # avg_loss, variance
    reconstructions.add(DISTANCE, results)
    reconstructions.save()


def mean_variance_samples():
    results = {}
    for noise in tqdm(NOISES_TESTING):
        if NOISE_MODEL == 'BitFlip':
            # data_test = BitFlipRotatedSurfaceData(distance=DISTANCE, noises=[noise],
            data_test = (BitFlipToricData(distance=DISTANCE, noises=[noise],
                                          name="DS_Testing-{0}".format(DISTANCE),
                                          load=False, random_flip=random_flip, sequential=sequential)
                         .eval()
                         .initialize(num=100))
            mean = torch.mean(data_test.syndromes, dim=(1, 2, 3))
            var = torch.var(data_test.syndromes, dim=(1, 2, 3))
            results[noise] = (mean, var)
        elif NOISE_MODEL == 'Depolarizing':
            data_test = (DepolarizingToricData(distance=DISTANCE, noises=[noise],
                                               name="DS_Testing-{0}".format(DISTANCE),
                                               load=False, random_flip=random_flip, sequential=sequential)
                         .eval()
                         .initialize(num=100))
            mean = torch.mean(data_test.syndromes, dim=(2, 3))
            var = torch.var(data_test.syndromes, dim=(2, 3))
            results[noise] = (mean, var)
    raw = Predictions(name="mean_variance_" + str(NOISE_MODEL).lower() + "_" + str(DISTANCE))
    raw.add(DISTANCE, results)
    raw.save()


if __name__ == "__main__":
    (random_flip, LR, NOISE_MODEL, NUM_EPOCHS, BATCH_SIZE, DATA_SIZE, DISTANCE, LOAD_DATA, SAVE_DATA, NOISES_TRAINING,
     NOISES_TESTING, RATIO, LATENT_DIMS, STRUCTURE, SUPERVISED, CLUSTER) = parameters()
    # s = sys.argv[1]
    s = 41
    s = int(s)

    DISTANCE = DISTANCES[s % 10 - 1]
    SUPERVISED = 20 < s < 40

    structures = ['standard', 'simple', 'skip', 'ising', 'transformer', 'vision-transformer', 'cnn']
    if s < 10:
        STRUCTURE = structures[1]
    elif 10 < s < 20:
        STRUCTURE = structures[3]
    elif 20 < s < 30:
        STRUCTURE = structures[-1]
    elif 30 < s < 40:
        STRUCTURE = structures[-2]
    else:
        STRUCTURE = structures[1]

    pretrained = STRUCTURE == 'vision-transformer'

    NOISE_MODEL = 'BitFlip' if s < 40 else 'Depolarizing'  # 'BitFlip',  'Depolarizing'

    if NOISE_MODEL == 'Depolarizing':
        if not SUPERVISED:
            noises_testing = np.array(
                list(map(lambda x: np.exp(-4 / x) / (1 / 3 + np.exp(-4 / x)), np.arange(0.02, 3, 0.02))))
            noises_training = np.array(
                list(map(lambda x: np.exp(-4 / x) / (1 / 3 + np.exp(-4 / x)), np.arange(0.02, 3, 0.02))))
        else:
            noises_testing = np.array(
                list(map(lambda x: np.exp(-4 / x) / (1 / 3 + np.exp(-4 / x)), np.arange(0.02, 3, 0.02))))
            noises_training = np.concatenate((np.array(list(map(lambda x: np.exp(-4 / x) / (1 / 3 + np.exp(-4 / x)),
                                                                np.arange(0.02, 0.6, 0.02)))),
                                              np.array(list(map(lambda x: np.exp(-4 / x) / (1 / 3 + np.exp(-4 / x)),
                                                                np.arange(1.4, 2.0, 0.02))))))
    else:
        if not SUPERVISED:
            noises_testing = np.array(
                list(map(lambda x: np.exp(-2 / x) / (1 + np.exp(-2 / x)), np.arange(0.02, 2, 0.02))))
            noises_training = np.array(
                list(map(lambda x: np.exp(-2 / x) / (1 + np.exp(-2 / x)), np.arange(0.02, 2, 0.02))))
        else:
            noises_testing = np.array(
                list(map(lambda x: np.exp(-2 / x) / (1 + np.exp(-2 / x)), np.arange(0.2, 2, 0.2))))
            noises_training = np.concatenate((np.array(list(map(lambda x: np.exp(-2 / x) / (1 + np.exp(-2 / x)),
                                                                np.arange(0.02, 0.6, 0.02)))),
                                              np.array(list(map(lambda x: np.exp(-2 / x) / (1 + np.exp(-2 / x)),
                                                                np.arange(1.4, 2.0, 0.02))))))

    sequential = (STRUCTURE == 'transformer')
    beta = 500

    name_data = str(NOISE_MODEL) + "_T_" + ("rf_" if random_flip else "") + ("sq_" if sequential else "") + (
        "labels_" if SUPERVISED else "") + "f2_1-{0}"
    name_dict_predictions = "predictions_" + str(NOISE_MODEL) + "_" + STRUCTURE + ("_rf_" if random_flip else "") + str(
        DISTANCE) + "_f2_1"
    name_NN = "net_NN_" + str(NOISE_MODEL) + "_" + STRUCTURE + "_dim" + str(LATENT_DIMS) + (
        "_rf_" if random_flip else "") + "f2_1-" + str(DISTANCE)
    name_dict_recon = "reconstruction_" + str(NOISE_MODEL) + "_" + STRUCTURE + (
        "_rf_" if random_flip else "") + str(DISTANCE) + "f2_1"
    name_dict_latent = "latents_" + str(NOISE_MODEL) + "_" + STRUCTURE + (
        "_rf_" if random_flip else "") + str(DISTANCE) + "f2_1"

    # train_model()
    latents()
    if not SUPERVISED:
        reconstructions()
