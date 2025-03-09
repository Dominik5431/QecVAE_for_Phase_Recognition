from datetime import datetime
import matplotlib.pyplot as plt
from parameters import parameters
import torch
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
from src.nn import ResultsWrapper
from src.error_code import DepolarizingToricCode
from src.nn.utils.loss import loss_func
from src.nn.utils.optimizer import make_optimizer
from src.nn.utils.functions import smooth

import numpy as np
import logging

task_description = {0: "Create data", 1: "Train network", 2: "Evaluate latent space",
                    20: "Evaluate reconstruction error"}

# Distances to evaluate VAE
distances = [7, 9, 11, 15, 21, 27, 33]

if __name__ == "__main__":
    logger = logging.getLogger('vae_threshold')
    logger.setLevel(level=logging.DEBUG)
    fh = logging.StreamHandler()
    fh_formatter = logging.Formatter('%(asctime)s %(levelname)s %(lineno)d:%(filename)s(%(process)d) - %(message)s')
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)

    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')

    ''' Hyperparameters '''

    (RANDOM_FLIP, LR, NOISE_MODEL, NUM_EPOCHS, BATCH_SIZE, DATA_SIZE, DISTANCE, LOAD_DATA, SAVE_DATA, NOISES_TRAINING,
     NOISES_TESTING, RATIO, LATENT_DIMS, STRUCTURE, CLUSTER) = parameters()

    if STRUCTURE == 'upsampling' or STRUCTURE == 'skip':
        assert DISTANCE in [5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45]
    elif STRUCTURE == 'conv-only':
        assert DISTANCE in [9, 17, 25, 33, 41, 49]

    sequential = False
    surface = False
    only_syndromes = True

    # Dictionary with hyperparameters for VAE structure based upon TraDE. Motivation: TraDE structure has already
    # been successfully applied for QEC purpose.
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
    # ff2: final data for thesis

    iteration = "ff2"

    name_data = str(NOISE_MODEL) + "_" + ("rf_" if RANDOM_FLIP else "") + iteration + "-" + str(DISTANCE)
    name_NN = "nn_" + str(NOISE_MODEL) + "_" + STRUCTURE + "_dim" + str(LATENT_DIMS) + (
        "_rf_" if RANDOM_FLIP else "") + iteration + "-" + str(DISTANCE)
    name_dict_recon = "reconstruction_" + str(NOISE_MODEL) + "_" + STRUCTURE + (
        "_rf_" if RANDOM_FLIP else "") + str(DISTANCE) + iteration
    name_dict_latent = "latents_" + str(NOISE_MODEL) + "_" + STRUCTURE + (
        "_rf_" if RANDOM_FLIP else "") + str(DISTANCE) + iteration

    task = 0

    if task in (0, 1, 2, 20) and not CLUSTER:
        # Writes task summary to external file.
        with open("config.txt", "a") as f:
            f.write(str(datetime.now()) + " rf: " + str(RANDOM_FLIP) + ", structure: " + str(STRUCTURE)
                    + ", noise: " + str(NOISE_MODEL) + "\n")
            f.write('Noises: Training ' + str(NOISES_TRAINING[0]) + " " + str(NOISES_TRAINING[-1])
                    + " " + str(len(NOISES_TRAINING)) + ", Testing: " + str(NOISES_TESTING[0]) + " "
                    + str(NOISES_TESTING[-1]) + " " + str(len(NOISES_TESTING)) + "\n")
            f.write("Learning: lr: " + str(LR) + ", epochs: " + str(NUM_EPOCHS) + ", sz: " + str(DATA_SIZE)
                    + ", distance: " + str(DISTANCE) + "\n")
            f.write("Task: " + str(task) + task_description[task] + "\n")

    if task == 0:  # Create data
        # Generates data samples and
        #
        # saves them as .pt file.
        print("Create data.")
        if surface:  # sequential data for transformer-based encoder/decoder
            (DepolarizingSurfaceData(distance=DISTANCE, noises=NOISES_TRAINING, name=name_data.format(DISTANCE),
                                     load=False, device=device)
             .initialize(num=DATA_SIZE)
             .save())
        else:
            if NOISE_MODEL == 'BitFlip':
                (BitFlipToricData(distance=DISTANCE, noises=NOISES_TRAINING, name=name_data.format(DISTANCE),
                                  load=False, random_flip=RANDOM_FLIP, device=device, sequential=sequential,
                                  cluster=CLUSTER, only_syndromes=only_syndromes)
                 .training()
                 .initialize(num=DATA_SIZE)
                 .save())
            elif NOISE_MODEL == 'Depolarizing':
                (DepolarizingToricData(distance=DISTANCE, noises=NOISES_TRAINING, name=name_data.format(DISTANCE),
                                       load=False, random_flip=RANDOM_FLIP, device=device, sequential=sequential,
                                       cluster=CLUSTER, only_syndromes=only_syndromes)
                 .training()
                 .initialize(num=DATA_SIZE)
                 .save())
    elif task == 1:  # Training the network
        # Manages the training of the network.
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
                                     load=LOAD_DATA, random_flip=RANDOM_FLIP, device=device, sequential=sequential,
                                     cluster=CLUSTER, only_syndromes=only_syndromes)
                    .training()
                    .initialize(num=DATA_SIZE)
                    .get_train_test_data(RATIO))
            elif NOISE_MODEL == 'Depolarizing':
                data_train, data_val = (DepolarizingToricData(distance=DISTANCE, noises=NOISES_TRAINING,
                                                              name=name_data.format(DISTANCE),
                                                              load=LOAD_DATA,
                                                              random_flip=RANDOM_FLIP,
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
            net = train(net, make_optimizer(LR), loss_func, NUM_EPOCHS, BATCH_SIZE, data_train, data_val)
    elif task == 2:  # Evaluating the latent space
        # Evaluates the model latent space.
        # Creates dictionary containing the latent variables for specified noise strengths.
        logging.debug("Evaluate latent space.")
        if surface:
            model = TraVAE(latent_dims=LATENT_DIMS, distance=DISTANCE, name=name_NN.format(DISTANCE), **trade_dict)
            model.load()
            latents = ResultsWrapper(name=name_dict_latent)
            latents.load()
            results = {}
            for noise in tqdm(NOISES_TESTING):
                data_test = (DepolarizingSurfaceData(distance=DISTANCE, noises=[noise], name='test_data',
                                                     load=False, device=device, only_syndromes=only_syndromes)
                             .initialize(num=50000))
                results[noise] = test_latent_space_TraVAE(model, data_test, device=device)
            latents.add(DISTANCE, results)
            latents.save()
            result = latents.get_dict()
        else:
            model = VariationalAutoencoder(LATENT_DIMS, DISTANCE, name_NN.format(DISTANCE), structure=STRUCTURE,
                                           noise=NOISE_MODEL, device=device, cluster=CLUSTER)
            model.load()
            # Use dictionary with noise value and return values to store return data from VAE while testing
            latents = ResultsWrapper(name=name_dict_latent, cluster=CLUSTER)
            latents.load()
            results = {}
            for noise in tqdm(NOISES_TESTING):
                data_test = None
                if NOISE_MODEL == 'BitFlip':
                    # data_test = BitFlipRotatedSurfaceData(distance=DISTANCE, noises=[noise],
                    data_test = (BitFlipToricData(distance=DISTANCE, noises=[noise],
                                                  name="BFS_Testing-{0}".format(DISTANCE),
                                                  load=False, random_flip=RANDOM_FLIP, sequential=sequential,
                                                  device=device, cluster=CLUSTER, only_syndromes=only_syndromes)
                                 .eval()
                                 .initialize(num=1000))
                elif NOISE_MODEL == 'Depolarizing':
                    data_test = (DepolarizingToricData(distance=DISTANCE, noises=[noise],
                                                       name="DS_Testing-{0}".format(DISTANCE),
                                                       load=False, random_flip=RANDOM_FLIP, sequential=sequential,
                                                       device=device, cluster=CLUSTER, only_syndromes=only_syndromes)
                                 .eval()
                                 .initialize(num=1000))
                assert data_test is not None
                # res = test_model_latent_space(model, data_test)

                m = torch.mean(data_test.syndromes, dim=(1, 2, 3))
                sus = (torch.mean(m ** 2) - torch.mean(torch.abs(m)) ** 2).cpu().detach().numpy()
                results[noise] = test_model_latent_space(model, data_test) + (
                    m, sus,)  # z_mean, z_log_var, z, flips, mean
            latents.add(DISTANCE, results)
            latents.save()
            result = latents.get_dict()

        plot_latent_susceptibility(result, RANDOM_FLIP, STRUCTURE, NOISE_MODEL, surface=surface)
    elif task == 20:  # Evaluate reconstruction loss
        # Evaluates the model reconstruction error for different noise strengths. Saves the reconstruction error as
        # dictionary.
        logging.debug("Evaluate reconstruction error.")
        model = VariationalAutoencoder(LATENT_DIMS, DISTANCE, name_NN.format(DISTANCE), structure=STRUCTURE,
                                       noise=NOISE_MODEL)
        model.load()
        # Use dictionary with noise value and return values to store return data from VAE while testing
        reconstructions = ResultsWrapper(name=name_dict_recon)
        reconstructions.load()
        results = {}
        for noise in tqdm(NOISES_TESTING):
            data_test = None
            if NOISE_MODEL == 'BitFlip':
                data_test = (BitFlipToricData(distance=DISTANCE, noises=[noise],
                                              name="BFS_Testing-{0}".format(DISTANCE),
                                              load=False, random_flip=RANDOM_FLIP, sequential=sequential)
                             .eval()
                             .initialize(num=100))
            elif NOISE_MODEL == 'Depolarizing':
                data_test = (DepolarizingToricData(distance=DISTANCE, noises=[noise],
                                                   name="DS_Testing-{0}".format(DISTANCE),
                                                   load=False, random_flip=RANDOM_FLIP, sequential=sequential)
                             .eval()
                             .initialize(num=100))
            results[noise] = test_model_reconstruction_error(model, data_test,
                                                             torch.nn.MSELoss(
                                                                 reduction='none'))  # returns avg_loss, variance
        reconstructions.add(DISTANCE, results)
        reconstructions.save()
    elif task == 3:  # Plot latent space, computed in task 2
        test = ResultsWrapper(name=name_dict_latent).load().get_dict()
        # plot_latent_mean(test, random_flip, STRUCTURE)
        if surface:
            plot_latent_susceptibility(test, RANDOM_FLIP, STRUCTURE, NOISE_MODEL, surface=surface)
        else:
            plot_latent_susceptibility(test, RANDOM_FLIP, STRUCTURE, NOISE_MODEL)
        # scatter_latent_var(test, random_flip, STRUCTURE)
        # plot_binder_cumulant(test, random_flip, STRUCTURE, NOISE_MODEL)
    elif task == 30:  # plot reconstruction error, computed in task 20
        recon = ResultsWrapper(name=name_dict_recon).load().get_dict()
        plot_reconstruction_error(recon, RANDOM_FLIP, STRUCTURE)
        plot_reconstruction_derivative(recon, RANDOM_FLIP, STRUCTURE)
    elif task == 4:  # Merge data dictionaries
        latents = ResultsWrapper(name="latents_" + str(NOISE_MODEL) + "_" + STRUCTURE + (
            "_rf_" if RANDOM_FLIP else "") + iteration)
        dists = [7, 9, 11, 17, 21]
        for i, dist in enumerate(dists):
            dictionary = ResultsWrapper(name="latents_" + str(NOISE_MODEL) + "_" + STRUCTURE + (
                "_rf_" if RANDOM_FLIP else "") + str(DISTANCE) + iteration).load().get_dict()
            latents.add(dist, dictionary)
        latents.save()
    elif task == 5:  # Plot exemplary reconstruction
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
    elif task == 6:  # Get mean and variance of raw data samples
        # Calculates mean and variance of the raw syndrome samples.
        results = {}
        # for noise in tqdm(NOISES_TESTING):
        for noise in np.arange(0.01, 0.2, 0.01):
            if NOISE_MODEL == 'BitFlip':
                data_test = (BitFlipToricData(distance=DISTANCE, noises=[noise],
                                              name="BFS_Testing-{0}".format(DISTANCE),
                                              load=False, random_flip=False, sequential=sequential,
                                              only_syndromes=True, device=device)
                             .eval()
                             .initialize(num=1000))
                # mean_tot = torch.mean(data_test.syndromes[0], dim=(0, 1, 2, 3))
                mean = torch.mean(data_test.syndromes[0], dim=(1, 2, 3))
                var = torch.var(data_test.syndromes[0], dim=(1, 2, 3))
                # print(mean_tot)
                # print(var)
                results[noise] = (mean, var)
            elif NOISE_MODEL == 'Depolarizing':
                data_test = (DepolarizingToricData(distance=DISTANCE, noises=[noise],
                                                   name="DS_Testing-{0}".format(DISTANCE),
                                                   load=False, random_flip=RANDOM_FLIP, sequential=sequential,
                                                   device=device)
                             .eval()
                             .initialize(num=100))
                mean_tot = torch.mean(data_test.syndromes[0], dim=(0, 1, 2, 3))
                mean = torch.mean(data_test.syndromes[0], dim=(1, 2, 3))
                var = torch.var(mean)
                results[noise] = (mean, var)
        raw = ResultsWrapper(name="mean_variance_" + str(NOISE_MODEL).lower() + "_2_" + str(DISTANCE))
        raw.add(DISTANCE, results)
        raw.save()
        raw = raw.get_dict()
        assert raw != {}

        plot_mean_variance_samples(raw, DISTANCE, NOISE_MODEL)
    elif task == 7:  # Perform data collapse
        logging.debug("Start data collapse.")

        # Load data to analyze
        predictions = ResultsWrapper(name_dict_latent).load().get_dict()
        # Plot.Plotter.plot_prediction(predictions, noises)

        # Calculate data collapse
        res = functions.data_collapse(NOISES_TESTING, predictions)
        pc = res.x[0]
        nu = res.x[1]
        # print(res.x)

        # Plotting
        plot_collapsed(predictions, NOISES_TESTING, pc, nu)
    elif task == 8:  # For testing: Creates exemplary data samples and computes forward pass.
        model = TraVAE(latent_dims=LATENT_DIMS, distance=DISTANCE, name=name_NN.format(DISTANCE), **trade_dict)
        model.load()
        model = model.to(device)
        noise = 0.1
        data_test = (DepolarizingSurfaceData(distance=DISTANCE, noises=[noise], name='test_data',
                                             load=False, device=device)
                     .initialize(num=10))
        recon, mean, log_var, z = model.forward(data_test.get_syndromes())
    elif task == 9:  # Show network params
        net = VariationalAutoencoder(LATENT_DIMS, DISTANCE, name_NN.format(DISTANCE), structure=STRUCTURE,
                                     noise=NOISE_MODEL)
        if STRUCTURE == 'transformer':
            summary(net, (DISTANCE ** 2, 1))
        else:
            summary(net, (1, 1, DISTANCE, DISTANCE))
    elif task == 10:  # Plot QEC code
        code = DepolarizingToricCode(3, 0.1, False)
        code.circuit_to_png()
    elif task == 11:  # Plot Ising data and Toric code data: Used for presentation and thesis
        # temperature = 0.9
        # noise = np.exp(-4 / temperature) / (1 / 3 + np.exp(-4 / temperature))
        DISTANCE = 29
        noise = 0.01

        # Generate exemplary sample
        sample = BitFlipToricData(distance=DISTANCE, noises=[noise],
                                  name="BFS_Testing-{0}".format(DISTANCE),
                                  load=False, random_flip=False,
                                  sequential=sequential, device=device,
                                  only_syndromes=only_syndromes).training().initialize(
            10)
        sample = sample[0]
        syndrome = sample[0][0].squeeze()
        print(syndrome.shape)

        # Plotting
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
    else:
        print("Unknown task number.")
        exit(-1)
