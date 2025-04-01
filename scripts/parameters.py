import numpy as np

"""
Set the hyperparameters.
"""


def parameters():
    random_flip = True  # Flips every second syndrome to assure Z2 symmetry in the data
    cluster = False  # Set True when executing on the HPC cluster

    # structures = ['standard', 'simple', 'skip', 'ising', 'transformer', 'vision-transformer', 'cnn']
    structures = ['conv-only', 'upsampling', 'skip']
    structure = structures[1]  # Select encoder-decoder structure

    lr = 1e-4
    noise_model = 'BitFlip'  # Implemented noise models: 'BitFlip',  'Depolarizing'
    assert noise_model in ['BitFlip', 'Depolarizing']

    # Training hyperparameters
    num_epochs = 10
    batch_size = 100
    distance = 25  # 25
    data_size = 8000 if distance > 30 else 10000

    # Load samples / Save generated samples
    load_data = True
    save_data = True

    # Train on noise strengths sampled uniformly along the Nishimori temperature line coming out from the
    # statistical mechanical mapping of the Toric code to some random bond Ising models.
    if noise_model == 'Depolarizing':
        noises_training = np.array(
           list(map(lambda x: np.exp(-4 / x) / (1 / 3 + np.exp(-4 / x)), np.arange(0.1, 3, 0.05))))
    else:
        noises_training = np.array(
            list(map(lambda x: np.exp(-2 / x) / (1 + np.exp(-2 / x)), np.arange(0.1, 2, 0.05))))

    # Data for evaluating latent space and reocnstruction error is set to be equal the training data
    if noise_model == 'Depolarizing':
        noises_testing = np.array(
            list(map(lambda x: np.exp(-4 / x) / (1 / 3 + np.exp(-4 / x)), np.arange(0.1, 3, 0.02))))
    else:
        noises_testing = np.array(
            list(map(lambda x: np.exp(-2 / x) / (1 + np.exp(-2 / x)), np.arange(0.1, 2, 0.02))))

    # Test/Val split
    ratio = 0.8
    # Dimension of the latent space
    latent_dims = 1

    return (random_flip, lr, noise_model, num_epochs, batch_size, data_size, distance, load_data, save_data,
            noises_training, noises_testing, ratio, latent_dims, structure, cluster)
