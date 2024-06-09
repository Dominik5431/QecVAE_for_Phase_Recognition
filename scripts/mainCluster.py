import sys
import torch
from src.nn.data import BitFlipSurfaceData
from src.nn.net import VariationalAutoencoder
from src.nn.data.predictions import Predictions
from src.nn.test import test_model_latent_space, test_model_reconstruction_error
from src.nn.train import train
from src.nn.utils.loss import loss_func
from src.nn.utils.optimizer import make_optimizer
from parameters import parameters
from tqdm import tqdm

NOISE_MODEL = 'BitFlip'
structures = ['standard', 'simple', 'skip', 'ising']
betas = [0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 5000, 10000]
distances = [15, 21, 27, 33, 37, 43]

def train_model():
    if NOISE_MODEL == 'BitFlip':
        # data_train, data_val = BitFlipRotatedSurfaceData(distance=DISTANCE, noises=NOISES_TRAINING,
        data_train, data_val = BitFlipSurfaceData(distance=DISTANCE, noises=NOISES_TRAINING,
                                                  name=name_data.format(DISTANCE),
                                                  num=DATA_SIZE, load=False,
                                                  random_flip=random_flip).get_train_test_data(RATIO)
    model = VariationalAutoencoder(LATENT_DIMS, DISTANCE, name_VAE.format(DISTANCE), structure=structure, noise=NOISE_MODEL)
    model = train(model, make_optimizer(LR), loss_func, NUM_EPOCHS, BATCH_SIZE, data_train, data_val, beta)

def latents():
    model = VariationalAutoencoder(LATENT_DIMS, DISTANCE, name_VAE.format(DISTANCE), structure=structure, noise=NOISE_MODEL)
    model.load()
    # Use dictionary with noise value and return values to store return data from VAE while testing
    latents = Predictions(name=name_dict_latent)
    latents.load()
    results = {}
    for noise in tqdm(NOISES_TESTING):
        # data_test = BitFlipRotatedSurfaceData(distance=DISTANCE, noises=[noise],
        data_test = BitFlipSurfaceData(distance=DISTANCE, noises=[noise],
                                       name="BFS_Testing-{0}".format(DISTANCE),
                                       num=1000, load=False, random_flip=random_flip)
        results[noise] = test_model_latent_space(model, data_test)  # z_mean, z_logvar, z, z_bar, z_bar_var
    latents.add(DISTANCE, results)
    latents.save()


def reconstructions():
    model = VariationalAutoencoder(LATENT_DIMS, DISTANCE, name_VAE.format(DISTANCE), structure=structure, noise=NOISE_MODEL)
    model.load()
    # Use dictionary with noise value and return values to store return data from VAE while testing
    reconstructions = Predictions(name=name_dict_recon)
    reconstructions.load()
    results = {}
    for noise in tqdm(NOISES_TESTING):
        # data_test = BitFlipRotatedSurfaceData(distance=DISTANCE, noises=[noise],
        data_test = BitFlipSurfaceData(distance=DISTANCE, noises=[noise],
                                       name="BFS_Testing-{0}".format(DISTANCE),
                                       num=1000, load=False, random_flip=random_flip)
        results[noise] = test_model_reconstruction_error(model, data_test, torch.nn.MSELoss())  # avg_loss
    reconstructions.add(DISTANCE, results)
    reconstructions.save()


if __name__ == "__main__":
    random_flip, LR, NOISE_MODEL, NUM_EPOCHS, BATCH_SIZE, DATA_SIZE, DISTANCE, LOAD_DATA, SAVE_DATA, NOISES_TRAINING, NOISES_TESTING, RATIO, LATENT_DIMS = parameters()
    # TODO add parser again
    # s = sys.argv[1]
    s = 3
    s = int(s)

    DISTANCE = distances[s]
    structure = structures[2]
    beta = betas[7]

    name_data = "BFS_2-{0}"
    name_dict_recon = "reconstruction_bitflip_" + structure + "_dim" + str(LATENT_DIMS) + "discrete" + str(DISTANCE)
    name_dict_latent = "latents_bitflip_" + structure + "_dim" + str(LATENT_DIMS) + "discrete" + str(DISTANCE)
    name_VAE = "VAE_" + structure + "_dim" + str(LATENT_DIMS) + "discrete-{0}"

    train_model()
    latents()
    reconstructions()
