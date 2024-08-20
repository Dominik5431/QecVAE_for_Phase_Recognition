from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from parameters import parameters
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchinfo import summary

import src.nn.utils.functions as functions
from src.nn import DepolarizingSurfaceData, BitFlipSurfaceData
from src.nn import VariationalAutoencoder
from src.nn.utils.plotter import plot_latent_mean, scatter_latent_var, plot_reconstruction_error, \
    plot_latent_susceptibility, plot_reconstruction_derivative, plot_reconstruction, plot_collapsed, \
    plot_mean_variance_samples, plot_binder_cumulant
from src.nn.train import train
from src.nn.test import test_model_latent_space, test_model_reconstruction_error
from src.nn import Predictions
from src.error_code import BitFlipSurfaceCode, DepolarizingSurfaceCode, SurfaceCodePheno
from src.nn.utils.loss import loss_func
from src.nn.utils.optimizer import make_optimizer
from src.nn.utils.functions import smooth

import numpy as np
import logging

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
task_description = {0: "Create data", 1: "Train network", 2: "Evaluate latent space",
                    20: "Evaluate reconstruction error"}


def prepare_data():
    logging.debug("Create data.")
    if NOISE_MODEL == 'BitFlip':
        # data = BitFlipRotatedSurfaceData(distance=DISTANCE, noises=NOISES_TRAINING, name=name_data.format(DISTANCE),
        (BitFlipSurfaceData(distance=DISTANCE, noises=NOISES_TRAINING, name=name_data.format(DISTANCE),
                            load=False, random_flip=random_flip, sequential=sequential,
                            supervised=False)  # Supervised per default False
         .training()
         .initialize(num=DATA_SIZE)
         .save())
    elif NOISE_MODEL == 'Depolarizing':
        (DepolarizingSurfaceData(distance=DISTANCE, noises=NOISES_TRAINING, name=name_data.format(DISTANCE),
                                 load=False, random_flip=random_flip, sequential=sequential)
         .training()
         .initialize(num=DATA_SIZE)
         .save())
    elif NOISE_MODEL == 'Phenomenological':
        data = None


def train_network():
    logging.debug("Get data.")
    data_train = None
    data_val = None
    if NOISE_MODEL == 'BitFlip':
        # data_train, data_val = BitFlipRotatedSurfaceData(distance=DISTANCE, noises=NOISES_TRAINING,
        data_train, data_val = (BitFlipSurfaceData(distance=DISTANCE, noises=NOISES_TRAINING,
                                                   name=name_data.format(DISTANCE),
                                                   load=LOAD_DATA,
                                                   random_flip=random_flip,
                                                   sequential=sequential,
                                                   supervised=False)
                                .training()
                                .initialize(num=DATA_SIZE)
                                .get_train_test_data(RATIO))
    elif NOISE_MODEL == 'Depolarizing':
        data_train, data_val = (DepolarizingSurfaceData(distance=DISTANCE, noises=NOISES_TRAINING,
                                                        name=name_data.format(DISTANCE),
                                                        load=LOAD_DATA,
                                                        random_flip=random_flip,
                                                        sequential=sequential)
                                .training()
                                .initialize(num=DATA_SIZE)
                                .get_train_test_data(RATIO))
    logging.debug("Train nn.")
    assert data_train is not None
    assert data_val is not None
    net = VariationalAutoencoder(LATENT_DIMS, DISTANCE, name_NN.format(DISTANCE), structure=STRUCTURE,
                                 noise=NOISE_MODEL)
    net = train(net, make_optimizer(LR), loss_func, NUM_EPOCHS, BATCH_SIZE, data_train, data_val)
    return net


def evaluate_latent_space():
    logging.debug("Evaluate latent space.")
    model = VariationalAutoencoder(LATENT_DIMS, DISTANCE, name_NN.format(DISTANCE), structure=STRUCTURE,
                                   noise=NOISE_MODEL)
    model.load()
    # Use dictionary with noise value and return values to store return data from VAE while testing
    latents = Predictions(name=name_dict_latent)
    latents.load()
    results = {}
    for noise in tqdm(NOISES_TESTING):
        data_test = None
        if NOISE_MODEL == 'BitFlip':
            # data_test = BitFlipRotatedSurfaceData(distance=DISTANCE, noises=[noise],
            data_test = (BitFlipSurfaceData(distance=DISTANCE, noises=[noise],
                                            name="BFS_Testing-{0}".format(DISTANCE),
                                            load=False, random_flip=random_flip, sequential=sequential)
                         .eval()
                         .initialize(num=100))
        elif NOISE_MODEL == 'Depolarizing':
            data_test = (DepolarizingSurfaceData(distance=DISTANCE, noises=[noise],
                                                 name="DS_Testing-{0}".format(DISTANCE),
                                                 load=False, random_flip=random_flip, sequential=sequential)
                         .eval()
                         .initialize(num=100))
        assert data_test is not None
        results[noise] = test_model_latent_space(model, data_test)  # z_mean, z_log_var, z, flips
    latents.add(DISTANCE, results)
    latents.save()


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
            data_test = (BitFlipSurfaceData(distance=DISTANCE, noises=[noise],
                                            name="BFS_Testing-{0}".format(DISTANCE),
                                            load=False, random_flip=random_flip, sequential=sequential)
                         .eval()
                         .initialize(num=100))
        elif NOISE_MODEL == 'Depolarizing':
            data_test = (DepolarizingSurfaceData(distance=DISTANCE, noises=[noise],
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
            data_test = (BitFlipSurfaceData(distance=DISTANCE, noises=[noise],
                                            name="BFS_Testing-{0}".format(DISTANCE),
                                            load=False, random_flip=random_flip, sequential=sequential)
                         .eval()
                         .initialize(num=10000))
            mean = torch.mean(data_test.syndromes, dim=(1, 2, 3))
            var = torch.var(data_test.syndromes, dim=(1, 2, 3))
            results[noise] = (mean, var)
        elif NOISE_MODEL == 'Depolarizing':
            data_test = (DepolarizingSurfaceData(distance=DISTANCE, noises=[noise],
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


# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

if __name__ == "__main__":
    logger = logging.getLogger('vae_threshold')
    logger.setLevel(level=logging.DEBUG)
    fh = logging.StreamHandler()
    fh_formatter = logging.Formatter('%(asctime)s %(levelname)s %(lineno)d:%(filename)s(%(process)d) - %(message)s')
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)
    # logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    # plt.set_loglevel("notset")

    #  = torch.device('mps')
    device = torch.device('cpu')

    (random_flip, LR, NOISE_MODEL, NUM_EPOCHS, BATCH_SIZE, DATA_SIZE, DISTANCE, LOAD_DATA, SAVE_DATA, NOISES_TRAINING,
     NOISES_TESTING, RATIO, LATENT_DIMS, STRUCTURE, SUPERVISED, CLUSTER) = parameters()

    sequential = STRUCTURE == 'transformer'

    name_data = str(NOISE_MODEL) + "_T_" + ("rf_" if random_flip else "") + ("sq_" if sequential else "") + (
        "labels_" if SUPERVISED else "") + "f3_2-{0}"
    name_NN = "net_NN_" + str(NOISE_MODEL) + "_" + STRUCTURE + "_dim" + str(LATENT_DIMS) + (
        "_rf_" if random_flip else "") + "f3_2-" + str(DISTANCE)
    name_dict_recon = "reconstruction_" + str(NOISE_MODEL) + "_" + STRUCTURE + (
        "_rf_" if random_flip else "") + str(DISTANCE) + "f3_2"
    name_dict_latent = "latents_" + str(NOISE_MODEL) + "_" + STRUCTURE + (
        "_rf_" if random_flip else "") + str(DISTANCE) + "f3_2"

    name_dict_latent = ("latents_BitFlip_ising_rf_f2_2")
    # name_dict_recon = "reconstruction_depolarizing_simple_dim1f8"
    name_NN = "net_NN_BitFlip_ising_dim1_rf_f2_2-{0}"

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

    if task == 0:  # Create data
        prepare_data()
    elif task == 1:  # Training a network
        train_network()
    elif task == 2:  # Testing a network
        evaluate_latent_space()
    elif task == 20:  # Test network via reconstruction loss
        evaluate_reconstruction_error()
    elif task == 3:  # plot latent space, computed in task 2
        test = Predictions(name=name_dict_latent).load().get_dict()
        # plot_latent_mean(test, random_flip, STRUCTURE)
        plot_latent_susceptibility(test, random_flip, STRUCTURE, NOISE_MODEL)
        # scatter_latent_var(test, random_flip, STRUCTURE)
        plot_binder_cumulant(test, random_flip, STRUCTURE, NOISE_MODEL)
    elif task == 30:  # plot reconstruction error, computed in task 20
        recon = Predictions(name=name_dict_recon).load().get_dict()
        plot_reconstruction_error(recon, random_flip, STRUCTURE)
        plot_reconstruction_derivative(recon, random_flip, STRUCTURE)
    elif task == 200:  # merge data dicts together
        latents = Predictions(name=name_dict_latent)
        # dists = [15, 21, 27, 33, 37, 43]
        dists = [27, 43]
        for i, dist in enumerate(dists):
            print(dist)
            print("latents_BitFlip_simple_rf_" + str(dist) + "f2_2.pkl")
            dictionary = Predictions(name="latents_BitFlip_ising_rf_" + str(dist) + "f2_2").load().get_dict()
            latents.add(dist, dictionary[dist])
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
        temperature = 0.7
        noise = np.exp(-2 / temperature) / (1 + np.exp(-2 / temperature))
        print(noise)
        sample = BitFlipSurfaceData(distance=DISTANCE, noises=[noise],
                                         name="BFS_Testing-{0}".format(DISTANCE),
                                         load=False, random_flip=random_flip,
                                         sequential=sequential).training().initialize(
            10)
        model = VariationalAutoencoder(LATENT_DIMS, DISTANCE, name_NN.format(DISTANCE), structure=STRUCTURE,
                                       noise=NOISE_MODEL)
        model = model.double().to(device)
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
    elif task == 4:  # Perform data collapse
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
                BitFlipSurfaceData(DISTANCE, [0.1], name='Dummy', num=1, load=False, random_flip=random_flip),
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
        code = SurfaceCodePheno(DISTANCE, 0.1, False)
        code.circuit_to_png()
    else:
        print("Unknown task number.")
        exit(-1)
