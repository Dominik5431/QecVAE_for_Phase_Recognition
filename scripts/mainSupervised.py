from datetime import datetime

from transformers import ViTFeatureExtractor, ViTImageProcessor

from parameters import parameters
import torch
from tqdm import tqdm

from src.nn import DepolarizingSurfaceData, BitFlipSurfaceData
from src.nn import VariationalAutoencoder
from src.nn.net.cnn import CNN

from src.nn.train import train_supervised
from src.nn.test import test_model_latent_space, test_model_reconstruction_error, test_model_predictions
from src.nn import Predictions
from src.nn.utils.optimizer import make_optimizer
from src.nn.utils.plotter import plot_predictions
from src.nn.net.vision_transformer import VisionTransformer

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
                            load=False, random_flip=random_flip, sequential=sequential, supervised=True)
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
    channels = -1
    if NOISE_MODEL == 'BitFlip':
        # data_train, data_val = BitFlipRotatedSurfaceData(distance=DISTANCE, noises=NOISES_TRAINING,
        data_train, data_val = (BitFlipSurfaceData(distance=DISTANCE, noises=NOISES_TRAINING,
                                                   name=name_data.format(DISTANCE),
                                                   load=LOAD_DATA,
                                                   random_flip=random_flip,
                                                   sequential=sequential,
                                                   supervised=True)
                                .training()
                                .initialize(num=DATA_SIZE)
                                .get_train_test_data(RATIO))
        channels = 1
    elif NOISE_MODEL == 'Depolarizing':
        data_train, data_val = (DepolarizingSurfaceData(distance=DISTANCE, noises=NOISES_TRAINING,
                                                        name=name_data.format(DISTANCE),
                                                        load=LOAD_DATA,
                                                        random_flip=random_flip,
                                                        sequential=sequential)
                                .training()
                                .initialize(num=DATA_SIZE)
                                .get_train_test_data(RATIO))
        channels = 2
    logging.debug("Train nn.")
    assert data_train is not None
    assert data_val is not None
    assert channels >= 1
    net = None
    if STRUCTURE == 'cnn':
        net = CNN(distance=DISTANCE, channels=channels, name=name_NN)
    elif STRUCTURE == 'vision-transformer':
        net = VisionTransformer()
    assert net is not None
    net = train_supervised(net, make_optimizer(LR), torch.nn.BCELoss(), NUM_EPOCHS, BATCH_SIZE, data_train, data_val)
    return net


def predict():
    logging.debug("Evaluate predictions.")
    channels = 0
    if NOISE_MODEL == 'BitFlip':
        channels = 1
    elif NOISE_MODEL == 'Depolarizing':
        channels = 2
    assert channels >= 1
    model = CNN(distance=DISTANCE, channels=channels, name=name_NN.format(DISTANCE))
    model.load()
    # Use dictionary with noise value and return values to store return data from VAE while testing
    preds = Predictions(name=name_dict_predictions)
    preds.load()
    results = {}
    for noise in tqdm(NOISES_TESTING):
        data_test = None
        if NOISE_MODEL == 'BitFlip':
            # data_test = BitFlipRotatedSurfaceData(distance=DISTANCE, noises=[noise],
            data_test = (BitFlipSurfaceData(distance=DISTANCE, noises=[noise],
                                            name="BFS_Testing-{0}".format(DISTANCE),
                                            load=False, random_flip=random_flip, sequential=sequential,
                                            supervised=True)
                         .eval()
                         .initialize(num=1000))
        elif NOISE_MODEL == 'Depolarizing':
            data_test = (DepolarizingSurfaceData(distance=DISTANCE, noises=[noise],
                                                 name="DS_Testing-{0}".format(DISTANCE),
                                                 load=False, random_flip=random_flip, sequential=sequential)
                         .eval()
                         .initialize(num=100))
        assert data_test is not None
        results[noise] = test_model_predictions(model, data_test)  # z
    preds.add(DISTANCE, results)
    preds.save()


if __name__ == "__main__":
    logger = logging.getLogger('vae_threshold')
    logger.setLevel(level=logging.DEBUG)
    fh = logging.StreamHandler()
    fh_formatter = logging.Formatter('%(asctime)s %(levelname)s %(lineno)d:%(filename)s(%(process)d) - %(message)s')
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)
    # logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    # plt.set_loglevel("notset")

    device = torch.device('mps')

    (random_flip, LR, NOISE_MODEL, NUM_EPOCHS, BATCH_SIZE, DATA_SIZE, DISTANCE, LOAD_DATA, SAVE_DATA, NOISES_TRAINING,
     NOISES_TESTING, RATIO, LATENT_DIMS, STRUCTURE, SUPERVISED, CLUSTER) = parameters()

    sequential = STRUCTURE == 'transformer'

    name_data = str(NOISE_MODEL) + "_T_" + ("rf_" if random_flip else "") + ("sq_" if sequential else "") + (
        "labels_" if SUPERVISED else "") + "TEST-{0}"
    name_dict_predictions = "predictions_" + str(NOISE_MODEL) + "_" + STRUCTURE + ("rf_" if random_flip else "") + str(
        DISTANCE) + "TEST"
    name_NN = "CNN_" + str(NOISE_MODEL) + "_" + STRUCTURE + "_dim" + str(LATENT_DIMS) + (
        "rf_" if random_flip else "") + "TEST-" + str(DISTANCE)

    name_dict_predictions = "predictions_BitFlip_cnn_rf_f2_2"

    task = 3

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
        predict()
    elif task == 3:  # Plot results
        test = Predictions(name=name_dict_predictions).load().get_dict()
        plot_predictions(test)
    elif task == 200:  # merge data dicts together
        predictions = Predictions(name=name_dict_predictions)
        dists = [15, 21, 27, 33, 37, 43]
        for i, dist in enumerate(dists):
            print(dist)
            print("predictions_BitFlip_cnn_rf_" + str(dist) + "f2_2")
            dictionary = Predictions(name="predictions_BitFlip_cnn_rf_" + str(dist) + "_f2_2").load().get_dict()
            predictions.add(dist, dictionary[dist])
        predictions.save()
    else:
        print('Invalid task number.')
        exit(-1)
