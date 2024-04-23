import argparse

from scipy.stats import sem
import NN
import Dataset
import numpy as np
import sys
import pickle
import gc
import Predictions


gc.enable()
layout = 3
load_model = True
ratio = 0.8
batch_size = 100
epochs = [10, 10, 7, 7, 4, 4]
filters = 10
rounds = 10

d_str = sys.argv[1]
d = int(d_str)

epochs = epochs[d]
distances = [15, 21, 27, 33, 39, 45]
distance = distances[d]
#noises = np.array([0.05, 0.20])
noises = np.array([0.12, 0.26]) # for depolarizing noise training
n = 100000


def collect_data(dist):
    name = "DS-{0}-lay{1}".format(dist, layout)
    dataset = Dataset.DepolarizingSurfaceData(dist, noises, name)
    dataset.generate_data(n, rounds)
    dataset.prepare_data(layout)
    train, val = dataset.get_training_data(ratio, batch_size)
    return train, val


def train_model():
    net = NN.CNNDepolarizing(distance, filters, layout)
    if load_model:
        try:
            net.load(layout)
        except Exception:
            pass
    for i in np.arange(5):
        train, val = collect_data(distance)
        history = net.train(train, val, epochs)
        with open('files/historyDepolarizing_d={0}_iter={1}.pkl'.format(distance, i), 'wb') as fp:
            pickle.dump(history, fp)
    net.save(layout)


def make_predictions(noise_min=0.01, noise_max=0.30, resolution=0.002, n_pred=50000):
    nn = NN.CNNDepolarizing(distance, filters, layout)
    nn.load(layout)
    noise_arr = np.arange(noise_min, noise_max, resolution)
    predics = np.zeros((len(noise_arr), 2))
    predics_err = np.zeros((len(noise_arr), 2))
    for k, noise in enumerate(noise_arr):
        syndromes = (Dataset.DepolarizingSurfaceData(distance, [noise], "pred_data")
                     .generate_data(n_pred, 1)
                     .prepare_data(layout)
                     .get_syndromes(batch_size))
        temp = nn.predict(syndromes)  # temp has shape (len(syndromes),2)
        del syndromes
        gc.collect()
        predics[k, :] = np.mean(temp, axis=0)
        predics_err[k, :] = sem(temp, axis=0)
    dictionary = Predictions.Predictions()
    dictionary.add(distance, (predics, predics_err))
    dictionary.save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, default=4, help="Task number")
    parser.add_argument("--noise", type=str, default="Depolarizing", help="Noise model")  # Bit-Flip vs Depolarizing
    parser.add_argument("--num_epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--data_size", type=int, default=3000, help="Dataset size")  # should be 10,000
    parser.add_argument("--batch_size", type=int, default=100, help="Training batch size")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate")
    parser.add_argument("--optimizer", type=str, default="Adam", help="Optimizer")
    parser.add_argument("--dist", type=int, default=33, help="Distance of QEC code")
    parser.add_argument("--load_data", type=bool, default=False, help="Specifies whether to generate the needed "
                                                                      "data or to load pregenerated dataset")
    parser.add_argument("--save_data", type=bool, default=False, help="Specifies whether to save used dataset")
    parsed, unparsed = parser.parse_known_args()
    # train_model()
    # make_predictions()
