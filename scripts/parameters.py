import numpy as np


def parameters():
    random_flip = True
    supervised = False
    cluster = False

    structures = ['standard', 'simple', 'skip', 'ising', 'transformer', 'vision-transformer', 'cnn']
    structure = structures[3]

    pretrained = structure == 'vision-transformer'

    lr = 0.001
    noise_model = 'BitFlip'  # 'BitFlip',  'Depolarizing'
    num_epochs = 100 if not pretrained else 10
    batch_size = 100
    data_size = 500
    distance = 27  # 15, 21, 27, 33, 39, 45
    load_data = True
    save_data = True

    # noises_training = np.arange(0, 0.30, 0.001)  # np.arange(0.05, 0.17, 0.001)  # for depolarizing noise: np.arange(0.10, 0.27, 0.01)
    # noises_testing = np.arange(0.0, 0.3, 0.003)
    # noises_training = np.concatenate((10**(np.arange(-3, -1, 0.004)), 10**np.arange(-1, -0.4, 0.002)))
    # noises_testing = 10**(np.arange(-3, -0.4, 0.005))
    # noises_training = np.array([0.005, 0.13])
    if noise_model == 'Depolarizing':
        if not supervised:
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
        if not supervised:
            noises_testing = np.array(
                list(map(lambda x: np.exp(-2 / x) / (1 + np.exp(-2 / x)), np.arange(0.02, 2, 0.02))))
            noises_training = np.array(
                list(map(lambda x: np.exp(-2 / x) / (1 + np.exp(-2 / x)), np.arange(0.02, 2, 0.02))))
        else:
            noises_testing = np.array(
                list(map(lambda x: np.exp(-2 / x) / (1 + np.exp(-2 / x)), np.arange(0.1, 3, 0.1))))
            noises_training = np.concatenate((np.array(list(map(lambda x: np.exp(-2 / x) / (1 + np.exp(-2 / x)),
                                                                np.arange(0.03, 0.6, 0.05)))),
                                              np.array(list(map(lambda x: np.exp(-2 / x) / (1 + np.exp(-2 / x)),
                                                                np.arange(1.4, 2.0, 0.05))))))
    # noises_training = np.array([0.01, 0.15])
    ratio = 0.8
    latent_dims = 1

    return (random_flip, lr, noise_model, num_epochs, batch_size, data_size, distance, load_data, save_data,
            noises_training, noises_testing, ratio, latent_dims, structure, supervised, cluster)
