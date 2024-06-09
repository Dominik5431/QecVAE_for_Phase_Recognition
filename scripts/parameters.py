import numpy as np


def parameters():
    random_flip = True
    lr = 0.0001
    noise_model = 'BitFlip'   # 'BitFlip',  'Depolarizing'
    num_epochs = 50
    batch_size = 100
    data_size = 500
    distance = 27  # 15, 21, 27, 33, 39, 45
    load_data = True
    save_data = False
    # noises_training = np.arange(0, 0.30, 0.001)  # np.arange(0.05, 0.17, 0.001)  # for depolarizing noise: np.arange(0.10, 0.27, 0.01)
    # noises_testing = np.arange(0.0, 0.3, 0.003)
    # noises_training = np.concatenate((10**(np.arange(-3, -1, 0.004)), 10**np.arange(-1, -0.4, 0.002)))
    # noises_testing = 10**(np.arange(-3, -0.4, 0.005))
    # noises_training = np.array([0.005, 0.13])
    noises_testing = np.array(list(map(lambda x: np.exp(-2/x)/(1+np.exp(-2/x)), np.arange(0.003, 3, 0.01))))
    noises_training = noises_testing
    ratio = 0.8
    latent_dims = 1
    return random_flip, lr, noise_model, num_epochs, batch_size, data_size, distance, load_data, save_data, noises_training, noises_testing, ratio, latent_dims
