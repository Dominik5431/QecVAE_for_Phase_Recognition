import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from src.nn import IsingData, VariationalAutoencoder, Predictions
from src.nn.test import test_model_latent_space, test_model_reconstruction_error
from src.nn.train import train
from src.nn.utils.loss import loss_func
from src.nn.utils.optimizer import make_optimizer
from src.nn.utils.plotter import plot_latent_mean, plot_latent_susceptibility, scatter_latent_var, \
    plot_reconstruction_error, plot_reconstruction_derivative

if __name__ == '__main__':
    task = 30

    # s = sys.argv[1]
    # s = int(s)

    L = 30
    n = 5
    T_max = 4
    LATENT_DIMS = 1
    LR = 0.001
    NUM_EPOCHS = 50
    BATCH_SIZE = 64
    beta = 500
    name_VAE = 'vae_ising-{0}'
    name_dict_latent = 'latents_ising'
    name_dict_recon = 'recons_ising'

    delta = 0.025
    if task == 1:
        data = IsingData(name="Ising", L=L, T_max=T_max, delta=delta).load()
        data_train, data_val, _ = data.get_train_test_data()
        assert data_train is not None
        assert data_val is not None
        model = VariationalAutoencoder(LATENT_DIMS, L, name_VAE.format(L), structure='ising',
                                       noise='Ising')
        model = train(model, make_optimizer(LR), loss_func, NUM_EPOCHS, BATCH_SIZE, data_train, data_val, beta)
    elif task == 2:
        data = IsingData(name="Ising", L=L, T_max=T_max, delta=delta).load()
        _, _, data_test = data.get_train_test_data()
        model = VariationalAutoencoder(LATENT_DIMS, L, name_VAE.format(L), structure='ising',
                                       noise='Ising')
        model.load()
        test_loader = DataLoader(data_test, batch_size=1)
        latents = Predictions(name=name_dict_latent)
        latents.load()
        results = {}
        temperatures = torch.arange(delta, T_max + delta, delta).__reversed__()
        for (batch_idx, batch) in enumerate(test_loader):
            print(batch_idx)
            results[temperatures[batch_idx]] = test_model_latent_space(
                model,
                IsingData('Test_data', L=L, T_max=T_max,
                          delta=delta,
                          configs=torch.permute(batch, [1, 0, 2, 3])))
        latents.add(L, results)
        latents.save()
    elif task == 20:
        data = IsingData(name="Ising", L=L, T_max=T_max, delta=delta).load()
        _, _, data_test = data.get_train_test_data()
        model = VariationalAutoencoder(LATENT_DIMS, L, name_VAE.format(L), structure='ising',
                                       noise='Ising')
        model.load()
        test_loader = DataLoader(data_test, batch_size=1)
        reconstructions = Predictions(name=name_dict_recon)
        reconstructions.load()
        results = {}
        temperatures = torch.arange(delta, T_max + delta, delta).__reversed__()
        for (batch_idx, batch) in enumerate(test_loader):
            results[temperatures[batch_idx]] = test_model_reconstruction_error(
                model,
                IsingData('Test_data', L=L, T_max=T_max,
                          delta=delta,
                          configs=torch.permute(batch, [1, 0, 2, 3])), torch.nn.MSELoss())
        reconstructions.add(L, results)
        reconstructions.save()
    elif task == 3:
        structure = 'ising'
        random_flip = True
        test = Predictions(name=name_dict_latent).load().get_dict()
        print(name_dict_latent)
        plot_latent_mean(test, random_flip, structure)
        plot_latent_susceptibility(test, random_flip, structure)
        scatter_latent_var(test, random_flip, structure)
    elif task == 30:
        structure = 'ising'
        random_flip = True
        recon = Predictions(name=name_dict_recon).load().get_dict()
        plot_reconstruction_error(recon, random_flip, structure)
        plot_reconstruction_derivative(recon, random_flip, structure)
    elif task == 100:
        data = IsingData(name="Ising", L=L, T_max=T_max, delta=delta).load()
        _, _, data_test = data.get_train_test_data()
        sample_high = data_test[3, 1]
        sample_low = data_test[110, 14]
        sample = sample_high
        model = VariationalAutoencoder(LATENT_DIMS, L, name_VAE.format(L), structure='ising',
                                       noise='Ising')
        model.load()
        model.eval()
        plt.imshow(sample.cpu().data.numpy(), cmap='magma')
        plt.colorbar()
        plt.show()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.double().to(device)
        with torch.no_grad():
            output, mean, log_var = model.forward(torch.reshape(sample, [1, 1, L, L]))
            # output = torch.where(output > 0.5, torch.ones_like(output), torch.zeros_like(output))
        plt.imshow(output.cpu().data.numpy()[0, 0], cmap='magma')
        plt.colorbar()
        plt.show()
