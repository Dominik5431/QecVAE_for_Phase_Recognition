import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from src.nn import IsingData, VariationalAutoencoder, Predictions
from src.nn.test import test_model_latent_space, test_model_reconstruction_error
from src.nn.train import train
from src.nn.utils.functions import smooth, simple_bootstrap
from src.nn.utils.loss import loss_func
from src.nn.utils.optimizer import make_optimizer
from src.nn.utils.plotter import plot_latent_mean, plot_latent_susceptibility, scatter_latent_var, \
    plot_reconstruction_error, plot_reconstruction_derivative

if __name__ == '__main__':
    task = 100

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
        latents = Predictions(name=name_dict_latent).load().get_dict()
        print(name_dict_latent)
        dists = list(latents.keys())
        coloring = ['black', 'blue', 'red', 'green', 'orange', 'pink', 'olive']
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        for k, dist in enumerate(dists):
            noises = list(latents[dist].keys())
            latent = list(latents[dist].values())

            means = list(map(lambda x: torch.mean(torch.abs(x[0])).cpu().detach().numpy(), latent))
            means = smooth(means, 5)
            unc = np.zeros(len(noises))
            for i in range(len(noises)):
                unc[i] = simple_bootstrap(latent[i][0], lambda x: torch.mean(torch.abs(x)).cpu().detach().numpy())
            # noises = np.array(list(map(lambda x: 4 * (1 - x) ** 3 * x + 4 * (1 - x) * x ** 3, noises)))
            # noises = np.array(list(map(lambda x: 2 / (np.log((1 - x) / x)), noises)))

            der = (np.array(list(map(lambda x: torch.mean(x[0] ** 2).cpu().detach().numpy(), latent))) -
                    np.array(list(map(lambda x: torch.mean(torch.abs(x[0])).cpu().detach().numpy() ** 2, latent))))[
                    1:]  # / noises[1:]
            unc2 = np.zeros(len(noises))
            for i in range(len(noises)):
                unc2[i] = simple_bootstrap(latent[i][0], lambda x: torch.mean(x ** 2).cpu().detach().numpy() - torch.mean(torch.abs(x)).cpu().detach().numpy() ** 2)
            ax1.plot(noises[1:], means[1:], color=coloring[k])
            ax1.errorbar(noises[1::3], means[1::3], yerr=unc[1::3], color=coloring[k], linestyle='dashed')
            ax2.plot(noises[1:], dist * der, color=coloring[k+1], label=str(dist))
            ax2.errorbar(noises[1:], dist * der, yerr=unc2[1:], color=coloring[k+1])
            # plt.vlines(1.373, 0, max(dist * der), colors='red', linestyles='dashed')
            # plt.vlines(0.95, 0, max(dist * der), colors='red', linestyles='dashed')
            # plt.vlines(0.109, 0, max(dist * der), colors='red', linestyles='dashed')


            # ax2.set_ylim([-1, 10])
            # plt.xlim(0, 2)


        plt.title("Structure: " + structure)
        ax1.tick_params(axis='y', labelcolor='black')
        # ax1.set_xlabel('associated temperature')
        ax1.set_xlabel('Temperature T')
        ax1.set_ylabel(r'mean $\langle \mu \rangle$')
        ax2.tick_params(axis='y', labelcolor='blue')
        ax2.set_ylabel('susceptibility', color='blue')
        plt.tight_layout()
        plt.legend()
        plt.show()

        # plot_latent_susceptibility(test, random_flip, structure)
        # scatter_latent_var(test, random_flip, structure)
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
