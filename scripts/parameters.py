import numpy as np


def parameters():
    random_flip = True
    supervised = False
    cluster = False

    structures = ['standard', 'simple', 'skip', 'ising', 'transformer', 'vision-transformer', 'cnn']
    structure = structures[0]

    lr = 0.0002
    noise_model = 'Depolarizing'  # 'BitFlip',  'Depolarizing'
    num_epochs = 100
    batch_size = 100
    data_size = 500
    distance = 17  # 15, 21, 27, 33, 39, 45
    load_data = True
    save_data = True

    '''
    if noise_model == 'Depolarizing':
        if not supervised:
            noises_testing = np.array(
                list(map(lambda x: np.exp(-4 / x) / (1 / 3 + np.exp(-4 / x)), np.arange(0.1, 2.5, 0.1))))
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
    '''
    # noises_training = np.arange(0.01, 0.4, 0.01)
    noises_training = np.array(list(map(lambda x: np.exp(-4 / x) / (1 / 3 + np.exp(-4 / x)), np.arange(0.02, 3, 0.02))))

    noises_testing = noises_training
    ratio = 0.8
    latent_dims = 3

    return (random_flip, lr, noise_model, num_epochs, batch_size, data_size, distance, load_data, save_data,
            noises_training, noises_testing, ratio, latent_dims, structure, supervised, cluster)
