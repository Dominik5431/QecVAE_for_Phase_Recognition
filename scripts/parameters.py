import numpy as np


def parameters():
    lr = 0.0001
    noise_model = 'BitFlip'  # alternative: 'Depolarizing'
    num_epochs = 100
    batch_size = 100
    data_size = 1000
    distance = 15  # 21, 27, 33, 39, 45
    load_data = True
    save_data = False
    noises_training = np.arange(0.05, 0.17, 0.001)  # for depolarizing noise: np.arange(0.10, 0.27, 0.01)
    noises_testing = np.arange(0.05, 0.17, 0.01)
    ratio = 0.8
    latent_dims = 1
    return lr, noise_model, num_epochs, batch_size, data_size, distance, load_data, save_data, noises_training, noises_testing, ratio, latent_dims

