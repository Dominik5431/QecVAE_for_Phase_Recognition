import os
import sys
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from PIL import Image
from scipy.optimize import curve_fit
from parameters import parameters
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchinfo import summary

import src.nn.utils.functions as functions
from src.nn import DepolarizingToricData, BitFlipToricData, DepolarizingSurfaceData
from src.nn import VariationalAutoencoder, TraVAE
from src.nn.utils.plotter import plot_reconstruction_error, \
    plot_latent_susceptibility, plot_reconstruction_derivative, plot_reconstruction, plot_collapsed, \
    plot_mean_variance_samples, plot_binder_cumulant, plot_correlation
from src.nn.train import train, train_TraVAE
from src.nn.test import test_model_latent_space, test_model_reconstruction_error, test_latent_space_TraVAE
from src.nn import Predictions
from src.error_code import DepolarizingToricCode
from src.nn.utils.loss import loss_func, loss_func_MSE
from src.nn.utils.optimizer import make_optimizer
from src.nn.utils.functions import smooth

import numpy as np
import logging

task_description = {0: "Create data", 1: "Train network", 2: "Evaluate latent space",
                    20: "Evaluate reconstruction error"}


def prepare_data():
    logging.debug("Create data.")
    if surface:
        (DepolarizingSurfaceData(distance=DISTANCE, noises=NOISES_TRAINING, name=name_data.format(DISTANCE),
                                 load=False, device=device)
         .initialize(num=DATA_SIZE)
         .save())
    else:
        if NOISE_MODEL == 'BitFlip':
            # data = BitFlipRotatedSurfaceData(distance=DISTANCE, noises=NOISES_TRAINING, name=name_data.format(DISTANCE),
            (BitFlipToricData(distance=DISTANCE, noises=NOISES_TRAINING, name=name_data.format(DISTANCE),
                              load=False, random_flip=random_flip, device=device, sequential=sequential,
                              cluster=CLUSTER, only_syndromes=only_syndromes,
                              supervised=False)  # Supervised per default False
             .training()
             .initialize(num=DATA_SIZE)
             .save())
        elif NOISE_MODEL == 'Depolarizing':
            (DepolarizingToricData(distance=DISTANCE, noises=NOISES_TRAINING, name=name_data.format(DISTANCE),
                                   load=False, random_flip=random_flip, device=device, sequential=sequential,
                                   cluster=CLUSTER, only_syndromes=only_syndromes)
             .training()
             .initialize(num=DATA_SIZE)
             .save())
        elif NOISE_MODEL == 'Phenomenological':
            pass


def train_network():
    logging.debug("Get data.")
    data_train = None
    data_val = None
    if surface:
        data_train, data_val = (
            DepolarizingSurfaceData(distance=DISTANCE, noises=NOISES_TRAINING, name=name_data.format(DISTANCE),
                                    load=True, device=device)
            .initialize(num=DATA_SIZE)
            .get_train_val_data(RATIO))
        net = TraVAE(latent_dims=LATENT_DIMS, distance=DISTANCE, name=name_NN.format(DISTANCE), **trade_dict)
        net = train_TraVAE(net, make_optimizer(LR), loss_func, NUM_EPOCHS, BATCH_SIZE, data_train, data_val)
    else:
        if NOISE_MODEL == 'BitFlip':
            # data_train, data_val = BitFlipRotatedSurfaceData(distance=DISTANCE, noises=NOISES_TRAINING,
            data_train, data_val = (
                BitFlipToricData(distance=DISTANCE, noises=NOISES_TRAINING, name=name_data.format(DISTANCE),
                                 load=LOAD_DATA, random_flip=random_flip, device=device, sequential=sequential,
                                 cluster=CLUSTER, only_syndromes=only_syndromes, supervised=False)
                .training()
                .initialize(num=DATA_SIZE)
                .get_train_test_data(RATIO))
        elif NOISE_MODEL == 'Depolarizing':
            data_train, data_val = (DepolarizingToricData(distance=DISTANCE, noises=NOISES_TRAINING,
                                                          name=name_data.format(DISTANCE),
                                                          load=LOAD_DATA,
                                                          random_flip=random_flip,
                                                          device=device,
                                                          sequential=sequential,
                                                          cluster=CLUSTER,
                                                          only_syndromes=only_syndromes)
                                    .training()
                                    .initialize(num=DATA_SIZE)
                                    .get_train_test_data(RATIO))
        logging.debug("Train nn.")
        assert data_train is not None
        assert data_val is not None
        net = VariationalAutoencoder(LATENT_DIMS, DISTANCE, name_NN.format(DISTANCE), structure=STRUCTURE,
                                     noise=NOISE_MODEL, device=device, cluster=CLUSTER)
        net = train(net, make_optimizer(LR), loss_func_MSE, NUM_EPOCHS, BATCH_SIZE, data_train, data_val)
    return net


def evaluate_latent_space():
    logging.debug("Evaluate latent space.")
    if surface:
        model = TraVAE(latent_dims=LATENT_DIMS, distance=DISTANCE, name=name_NN.format(DISTANCE), **trade_dict)
        model.load()
        latents = Predictions(name=name_dict_latent)
        latents.load()
        results = {}
        for noise in tqdm(NOISES_TESTING):
            data_test = (DepolarizingSurfaceData(distance=DISTANCE, noises=[noise], name='test_data',
                                                 load=False, device=device, only_syndromes=only_syndromes)
                         .initialize(num=50000))
            results[noise] = test_latent_space_TraVAE(model, data_test, device=device)
        latents.add(DISTANCE, results)
        latents.save()
        return latents.get_dict()
    else:
        model = VariationalAutoencoder(LATENT_DIMS, DISTANCE, name_NN.format(DISTANCE), structure=STRUCTURE,
                                       noise=NOISE_MODEL, device=device, cluster=CLUSTER)
        model.load()
        # Use dictionary with noise value and return values to store return data from VAE while testing
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
                                              device=device, cluster=CLUSTER, only_syndromes=only_syndromes)
                             .eval()
                             .initialize(num=500))
            elif NOISE_MODEL == 'Depolarizing':
                data_test = (DepolarizingToricData(distance=DISTANCE, noises=[noise],
                                                   name="DS_Testing-{0}".format(DISTANCE),
                                                   load=False, random_flip=random_flip, sequential=sequential,
                                                   device=device, cluster=CLUSTER, only_syndromes=only_syndromes)
                             .eval()
                             .initialize(num=500))
            assert data_test is not None
            # res = test_model_latent_space(model, data_test)

            m = torch.mean(data_test.syndromes, dim=(1, 2, 3))
            sus = (torch.mean(m ** 2) - torch.mean(torch.abs(m)) ** 2).cpu().detach().numpy()
            results[noise] = test_model_latent_space(model, data_test) + (m, sus,)  # z_mean, z_log_var, z, flips, mean
        latents.add(DISTANCE, results)
        latents.save()
        return latents.get_dict()


def evaluate_reconstruction_error():
    logging.debug("Evaluate reconstruction error.")
    model = VariationalAutoencoder(LATENT_DIMS, DISTANCE, name_NN.format(DISTANCE), structure=STRUCTURE,
                                   noise=NOISE_MODEL)
    model.load()
    # Use dictionary with noise value and return values to store return data from VAE while testing
    reconstructions = Predictions(name=name_dict_recon)
    reconstructions.load()
    results = {}
    for noise in tqdm(NOISES_TESTING):
        data_test = None
        if NOISE_MODEL == 'BitFlip':
            # data_test = BitFlipRotatedSurfaceData(distance=DISTANCE, noises=[noise],
            data_test = (BitFlipToricData(distance=DISTANCE, noises=[noise],
                                          name="BFS_Testing-{0}".format(DISTANCE),
                                          load=False, random_flip=random_flip, sequential=sequential)
                         .eval()
                         .initialize(num=100))
        elif NOISE_MODEL == 'Depolarizing':
            data_test = (DepolarizingToricData(distance=DISTANCE, noises=[noise],
                                               name="DS_Testing-{0}".format(DISTANCE),
                                               load=False, random_flip=random_flip, sequential=sequential)
                         .eval()
                         .initialize(num=100))
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
                                          name="BFS_Testing-{0}".format(DISTANCE),
                                          load=False, random_flip=random_flip, sequential=sequential)
                         .eval()
                         .initialize(num=10000))
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
    raw = Predictions(name="mean_variance_" + str(NOISE_MODEL).lower() + "_2_" + str(DISTANCE))
    raw.add(DISTANCE, results)
    raw.save()


# s = sys.argv[1]
# s = 0
# s = int(s)
distances = [7, 9, 11, 15, 21, 27, 33]

if __name__ == "__main__":
    logger = logging.getLogger('vae_threshold')
    logger.setLevel(level=logging.DEBUG)
    fh = logging.StreamHandler()
    fh_formatter = logging.Formatter('%(asctime)s %(levelname)s %(lineno)d:%(filename)s(%(process)d) - %(message)s')
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)
    # logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    # plt.set_loglevel("notset")

    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')

    ''' Hyperparameters '''

    (random_flip, LR, NOISE_MODEL, NUM_EPOCHS, BATCH_SIZE, DATA_SIZE, DISTANCE, LOAD_DATA, SAVE_DATA, NOISES_TRAINING,
     NOISES_TESTING, RATIO, LATENT_DIMS, STRUCTURE, SUPERVISED, CLUSTER) = parameters()

    # DISTANCE = distances[s]
    # NOISE_MODEL = 'BitFlip' if s < 7 else 'Depolarizing'

    sequential = False
    surface = False
    only_syndromes = True

    trade_dict = {
        'n': DISTANCE ** 2,
        'k': 1,
        'd_model': 4,
        'd_ff': 10,
        'n_layers': 1,
        'n_heads': 2,
        'device': device,
        'dropout': 0.2,
        'vocab_size': 2,
        'max_seq_len': 50,
    }

    ''' Version log '''

    # r2: latent var as start tokens
    # r3: generate tgt from latent dim, duplicated lat vectors
    # r4: back to idea from r2
    # r5: back to log data
    # r6: with standard structure, avgpooling
    # r7: TraVAE again with less params
    # r8: smaller dataset
    # r9: smaller TraVAE
    # r10: avg pool with logicals
    # r11: reduced complexity of model

    # ff1: final round for mid-term presentation

    name_data = str(NOISE_MODEL) + "_" + ("rf_" if random_flip else "") + ("sq_" if sequential else "") + (
        "labels_" if SUPERVISED else "") + "f2-" + str(DISTANCE)
    name_NN = "net_NN_" + str(NOISE_MODEL) + "_" + STRUCTURE + "_dim" + str(LATENT_DIMS) + (
        "_rf_" if random_flip else "") + "f2-" + str(DISTANCE)
    name_dict_recon = "reconstruction_" + str(NOISE_MODEL) + "_" + STRUCTURE + (
        "_rf_" if random_flip else "") + str(DISTANCE) + "f2"
    name_dict_latent = "latents_" + str(NOISE_MODEL) + "_" + STRUCTURE + (
        "_rf_" if random_flip else "") + str(DISTANCE) + "f2"

    # name_dict_latent = "latents_bitflip_simple_dim1f7"
    # name_dict_latent = "latents_Depolarizing_standard_rf_r7"
    # name_dict_recon = "reconstruction_depolarizing_simple_dim1f8"
    name_NN = "net_NN_BitFlip_ising_dim1_rf_f2_2-27"

    task = 100

    if task in (0, 1, 2, 20) and not CLUSTER:
        with open("config.txt", "a") as f:
            f.write(str(datetime.now()) + " rf: " + str(random_flip) + ", structure: " + str(STRUCTURE)
                    + ", supervised: " + str(SUPERVISED) + ", noise: " + str(NOISE_MODEL) + "\n")
            f.write('Noises: Training ' + str(NOISES_TRAINING[0]) + " " + str(NOISES_TRAINING[-1])
                    + " " + str(len(NOISES_TRAINING)) + ", Testing: " + str(NOISES_TESTING[0]) + " "
                    + str(NOISES_TESTING[-1]) + " " + str(len(NOISES_TESTING)) + "\n")
            f.write("Learning: lr: " + str(LR) + ", epochs: " + str(NUM_EPOCHS) + ", sz: " + str(DATA_SIZE)
                    + ", distance: " + str(DISTANCE) + "\n")
            f.write("Task: " + str(task) + task_description[task] + "\n")
    if task == -10:
        print(os.getcwd())
        plot_correlation()
    if task == -100:  # temp to plot Ising data and Toric code data
        # temperature = 0.9
        # noise = np.exp(-4 / temperature) / (1 / 3 + np.exp(-4 / temperature))
        DISTANCE = 29
        noise = 0.01
        sample = BitFlipToricData(distance=DISTANCE, noises=[noise],
                                  name="BFS_Testing-{0}".format(DISTANCE),
                                  load=False, random_flip=False,
                                  sequential=sequential, device=device,
                                  only_syndromes=only_syndromes).training().initialize(
            10)
        sample = sample[0]
        syndrome = sample[0][0].squeeze()
        print(syndrome.shape)

        '''
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        im = axs[0].imshow(np.reshape(syndrome[0].cpu().numpy(), (DISTANCE, DISTANCE)), cmap='magma')
        plt.colorbar(im)
        im = axs[1].imshow(np.reshape(syndrome[1].cpu().numpy(), (DISTANCE, DISTANCE)), cmap='magma')
        '''
        '''
        DISTANCE = 29

        image_path = 'Ising_lattice_2.png'  # Replace with your image path
        img = Image.open(image_path)

        # Convert the image to grayscale for easier pixel analysis
        img_gray = img.convert('L')
        pixel_values = np.array(img_gray)

        # Normalize pixel values between 0 and 1 for easier thresholding (0: black, 255: yellow)
        normalized_pixels = pixel_values / 255.0

        # Define thresholds for classification: black (close to 0) as +1, yellow (close to 1) as -1
        threshold = 0.5
        processed_pixels = np.where(normalized_pixels < threshold, 1, -1)

        # Resize the processed pixel data to a 29x29 grid using nearest neighbor method
        lattice_size = (29, 29)
        lattice_image = Image.fromarray((processed_pixels * 127.5 + 127.5).astype(np.uint8)).resize(lattice_size,
                                                                                                    Image.NEAREST)

        # Convert the resized image back to a numpy array and map back to +1 and -1
        lattice_array = np.where(np.array(lattice_image) < 128, 1, -1)
        syndrome = -1 * torch.as_tensor(lattice_array)
        '''

        fig, ax = plt.subplots()
        im = ax.imshow(np.reshape(syndrome.cpu().numpy(), (DISTANCE, DISTANCE)), cmap='inferno', origin='upper',
                       vmin=-1, vmax=1)

        ax.set_title(r'$d=29$')
        bar = fig.colorbar(im, ticks=[-1, 1])
        ax.set_xticks([])
        ax.set_yticks([])
        im.axes.get_xaxis().set_visible(False)
        im.axes.get_yaxis().set_visible(False)

        # Adjust the axis limits to center the image
        padding = 1  # Adjust this value to control how much space you want around the image
        ax.set_xlim(-padding, DISTANCE)
        ax.set_ylim(DISTANCE, -padding)  # Flip the y-axis to keep origin='upper'

        # Draw lines that extend beyond the image
        for i in range(-padding, DISTANCE + 1):
            ax.vlines(i - 0.5, -padding, DISTANCE, colors='black')
            ax.hlines(i - 0.5, -padding, DISTANCE, colors='black')

        plt.tight_layout()
        plt.savefig('syndrome_Toric_low_T.svg')
        plt.show()
    if task == 0:  # Create data
        prepare_data()
    elif task == 1:  # Training a network
        train_network()
    elif task == 2:  # Testing a network
        test = evaluate_latent_space()
        plot_latent_susceptibility(test, random_flip, STRUCTURE, NOISE_MODEL, surface=surface)
    elif task == 20:  # Test network via reconstruction loss
        evaluate_reconstruction_error()
    elif task == 3:  # plot latent space, computed in task 2
        test = Predictions(name=name_dict_latent).load().get_dict()
        # plot_latent_mean(test, random_flip, STRUCTURE)
        if surface:
            plot_latent_susceptibility(test, random_flip, STRUCTURE, NOISE_MODEL, surface=surface)
        else:
            plot_latent_susceptibility(test, random_flip, STRUCTURE, NOISE_MODEL)
        # scatter_latent_var(test, random_flip, STRUCTURE)
        # plot_binder_cumulant(test, random_flip, STRUCTURE, NOISE_MODEL)
    elif task == 4:
        model = TraVAE(latent_dims=LATENT_DIMS, distance=DISTANCE, name=name_NN.format(DISTANCE), **trade_dict)
        model.load()
        model = model.to(device)
        noise = 0.1
        data_test = (DepolarizingSurfaceData(distance=DISTANCE, noises=[noise], name='test_data',
                                             load=False, device=device)
                     .initialize(num=10))
        recon, mean, logvar, z = model.forward(data_test.get_syndromes())
        for name, param in model.named_parameters():
            print(name, param)
        print(recon)
        print(data_test.get_syndromes())
        print(mean)
        print(logvar)
    elif task == 30:  # plot reconstruction error, computed in task 20
        recon = Predictions(name=name_dict_recon).load().get_dict()
        plot_reconstruction_error(recon, random_flip, STRUCTURE)
        plot_reconstruction_derivative(recon, random_flip, STRUCTURE)
    elif task == 200:  # merge data dicts together
        latents = Predictions(name="latents_Depolarizing_standard_rf_r7")
        dists = [7, 9, 11, 17, 21]
        # dists = [27, 43]
        for i, dist in enumerate(dists):
            print(dist)
            print("latents_Depolarizing_standard_rf_" + str(dist) + "r7.pkl")
            dictionary = Predictions(name="latents_Depolarizing_standard_rf_" + str(dist) + "r7").load().get_dict()
            latents.add(dist, dictionary)
        latents.save()
    elif task == 31:
        dists = [7, 9, 15, 21, 27]
        # dists = [15, 21, 27, 33, 37, 43]

        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        coloring = ['black', 'blue', 'red', 'green', 'orange', 'pink']
        for i, dist in enumerate(dists):
            print(dist)
            latents = Predictions(name="latents_bitflip_skip_dim1f4-" + str(dist)).load().get_dict()
            noises = list(latents[dist].keys())
            noises = np.array(list(map(lambda x: 2 / (np.log((1 - x) / x)), noises)))
            latent = list(latents[dist].values())
            # print(latent[0][3].cpu().detach().numpy())
            # if latent[0][3].cpu().detach().numpy().ndim == 0:
            if len(latent[0][0][0].cpu().detach().numpy()) == 1:
                means = np.array(list(map(lambda x: torch.mean(torch.abs(x[0])).cpu().detach().numpy(), latent)))
                ax1.plot(noises, means, label=str(dist), color=coloring[i])
                print(dist)
                zs = np.array(list(map(lambda x: x[0].cpu().detach().numpy(), latent)))[:, :, 0]
                # print(np.shape(zs))
                flips = np.array(list(map(lambda x: x[3].cpu().detach().numpy(), latent)))

                der = np.zeros(len(noises))
                means = np.zeros(len(noises))
                for j in range(1, len(noises)):
                    vals = zs[j][np.where(flips[j] == -1)]
                    means[j] = np.mean(vals)
                    der[j] = np.mean(vals ** 2) - np.mean(vals) ** 2
                der = smooth(der, 7)
                ax2.plot(noises[1:], der[1:], label=str(dist), color=coloring[i])
                # plt.ylim([-0.00001, 0.00001])
                # ax2.set_ylim(0, 5)
                ax1.set_xlabel('associated temperature')
                ax1.set_ylabel(r'mean $\langle | \mu | \rangle$')
                ax2.set_ylabel(r' susceptibility $d \cdot \chi$')
                plt.vlines(0.95, 0, max(der), colors='red', linestyles='dashed')
                plt.xlim(0, 3)
                # plt.xscale('log')
                if random_flip:
                    plt.title(STRUCTURE + " + random flip")
                else:
                    plt.title(STRUCTURE)
        plt.legend()
        plt.tight_layout()
        plt.show()
    elif task == 32:
        dists = [7, 9, 15, 21, 27]
        # dists = [15, 21, 27, 33, 37, 43]
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        coloring = ['black', 'blue', 'red', 'green', 'orange', 'pink']
        for i, dist in enumerate(dists):
            reconstructions = Predictions(
                name="final3/reconstruction_bitflip_skip_dim1f3-" + str(dist)).load().get_dict()
            noises = list(reconstructions[dist].keys())
            noises = np.array(list(map(lambda x: 2 / (np.log((1 - x) / x)), noises)))
            reconstruction = list(reconstructions[dist].values())
            # print(latent[0][3].cpu().detach().numpy())
            # if latent[0][3].cpu().detach().numpy().ndim == 0:

            means = np.array(list(map(lambda x: x[0].cpu().detach().numpy(), reconstruction)))
            ax1.plot(noises, means, label=str(dist), color=coloring[i])
            means = smooth(means, 17)
            print(dist)
            der = np.gradient(means, noises)
            der = smooth(der, 13)
            ax2.plot(noises[10:], der[10:], label=str(dist), color=coloring[i])
            # plt.ylim([-0.00001, 0.00001])
            ax1.set_xlabel('associated temperature')
            ax1.set_ylabel(r'mean reconstruction error')
            ax2.set_ylabel(r'derivative')
            plt.vlines(0.95, -0.075, 0.18, colors='red', linestyles='dashed')
            plt.xlim(0, 2)
            ax2.set_ylim(-0.075, 0.15)
            # plt.xscale('log')
            if random_flip:
                plt.title(STRUCTURE + " + random flip")
            else:
                plt.title(STRUCTURE)
        plt.legend()
        fig.tight_layout()
        plt.show()
    elif task == 33:
        latents = Predictions(name=name_dict_latent).load().get_dict()
        dists = list(latents.keys())
        for dist in dists:
            noises = list(latents[dist].keys())
            latent = list(latents[dist].values())
            if len(latent[0][3].cpu()) == 1:
                p1idx = 5
                p2idx = -5
                p1 = noises[5]
                p2 = noises[-5]
                plt.hist(latent[p1idx][2].cpu().detach().numpy(), bins=50)
                plt.show()
                plt.hist(latent[p2idx][2].cpu().detach().numpy(), bins=50)
                plt.show()
    elif task == 100:
        # noise = 0.15
        # temperature = 0.9
        # noise = np.exp(-4 / temperature) / (1 / 3 + np.exp(-4 / temperature))
        noise = 0.109
        print(noise)
        sample = BitFlipToricData(distance=DISTANCE, noises=[noise],
                                       name="BFS_Testing-{0}".format(DISTANCE),
                                       load=False, random_flip=False,
                                       sequential=sequential, device=device,
                                       only_syndromes=only_syndromes).training().initialize(
            10)
        model = VariationalAutoencoder(LATENT_DIMS, DISTANCE, name_NN.format(DISTANCE), structure=STRUCTURE,
                                       noise=NOISE_MODEL, device=device)
        model = model.to(device)
        plot_reconstruction(sample, noise, DISTANCE, model)
    elif task == 101:  # Mean and variance of data
        mean_variance_samples()
    elif task == 102:
        raw = Predictions(name="mean_variance_" + str(NOISE_MODEL).lower() + "_2_" + str(DISTANCE))
        raw.load()
        raw = raw.get_dict()
        assert raw != {}
        # plot_latent_mean(raw, random_flip, STRUCTURE)
        plot_mean_variance_samples(raw, DISTANCE, NOISE_MODEL)
        # scatter_latent_var(raw, random_flip, STRUCTURE)
    elif task == 103:
        model = VariationalAutoencoder(LATENT_DIMS, DISTANCE, name_VAE.format(DISTANCE), structure=STRUCTURE,
                                       noise=NOISE_MODEL)
        model = model.double().to(device)
        model.load()
        model.eval()
        z = 1
        output = model.decoder.forward(z)
        plt.imshow(output)
        plt.show()
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
    elif task == 40:  # Perform data collapse
        logging.debug("Start data collapse.")
        predictions = Predictions(name_dict_latent).load().get_dict()
        # Plot.Plotter.plot_prediction(predictions, noises)
        res = functions.data_collapse(NOISES_TESTING, predictions)
        pc = res.x[0]
        nu = res.x[1]
        print(res.x)
        plot_collapsed(predictions, NOISES_TESTING, pc, nu)
    elif task == 6:  # Show network STRUCTURE
        if NOISE_MODEL == 'BitFlip':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = VariationalAutoencoder(LATENT_DIMS, DISTANCE, name_VAE.format(DISTANCE),
                                           structure=STRUCTURE).double().to(device)
            dummy = next(iter(DataLoader(
                BitFlipToricData(DISTANCE, [0.1], name='Dummy', num=1, load=False, random_flip=random_flip),
                batch_size=1, shuffle=False)))
            input_names = ['Syndrome']
            output_names = ['Reconstruction']
            torch.onnx.export(model, dummy, str(Path().resolve().parent) + "/data/vae.onnx", input_names=input_names,
                              output_names=output_names, verbose=True)
        elif NOISE_MODEL == 'Depolarizing':
            pass
    elif task == 7:  # Show network params
        net = VariationalAutoencoder(LATENT_DIMS, DISTANCE, name_VAE.format(DISTANCE), structure=STRUCTURE,
                                     noise=NOISE_MODEL)
        if STRUCTURE == 'transformer':
            summary(net, (DISTANCE ** 2, 1))
        else:
            summary(net, (1, 1, DISTANCE, DISTANCE))
    elif task == 8:  # Plot QEC code
        code = DepolarizingToricCode(3, 0.1, False)
        code.circuit_to_png()
    else:
        print("Unknown task number.")
        exit(-1)
