from pathlib import Path

import matplotlib.pyplot as plt
import scipy
from matplotlib import gridspec
import numpy as np
from src.nn.utils.functions import smooth
import seaborn as sns
from tqdm import tqdm
import torch.nn.functional as F
import torch
from scipy.optimize import curve_fit

from src.nn.utils.loss import loss_func


def plot_latent_mean(latents: dict, random_flip: bool, structure: str):
    dists = list(latents.keys())
    for dist in dists:
        noises = list(latents[dist].keys())
        noises = np.array(list(map(lambda x: 2 / (np.log((1 - x) / x)), noises)))
        latent = list(latents[dist].values())
        # print(latent[0][3].cpu().detach().numpy())
        # if latent[0][3].cpu().detach().numpy().ndim == 0:
        if len(latent[0][3].cpu().detach().numpy()) == 1:
            for j, noise in enumerate(noises):
                if not random_flip:
                    plt.scatter(noise, latent[j][3].cpu(), s=5, color='black')
                else:
                    mu_mean = torch.mean(torch.abs(latent[j][0]), dim=0).cpu().detach().numpy()
                    plt.scatter(noise, mu_mean, s=5, color='black')
            #plt.ylim([-0.00001, 0.00001])
            plt.xlabel('bitflip probability p')
            plt.ylabel(r'mean $\langle | \mu | \rangle$')
            # plt.xscale('log')
            if random_flip:
                plt.title(structure + " + random flip")
            else:
                plt.title(structure)
            plt.show()
            for j, noise in enumerate(noises):
                plt.scatter(noise, latent[j][4].cpu(), s=5, color='black')
                # if random_flip:
                # plt.scatter(noise, latent[j][4].cpu()/torch.mean(torch.abs(latent[j][0]), dim=0).cpu(), s=5, color='green')
            #plt.ylim([-0.00001, 0.00001])
            plt.xlabel('bitflip probability p')
            plt.ylabel(r'mean $\langle \sigma^2 \rangle$')
            # plt.xscale('log')
            if random_flip:
                plt.title(structure + " + random flip")
            else:
                plt.title(structure)
            plt.show()
        # elif latent[0][3].cpu().detach().numpy().ndim == 1:
        elif len(latent[0][3].cpu().detach().numpy()) == 2:
            cmap = sns.cubehelix_palette(as_cmap=True)
            f, ax = plt.subplots()
            x = []
            y = []
            z = []
            for j, noise in tqdm(enumerate(noises)):
                for i in np.arange(0, len(latent[j][2].cpu()), 100):
                    x.append(latent[j][2][i, 0].cpu().detach().numpy())
                    y.append(latent[j][2][i, 1].cpu().detach().numpy())
                    z.append(noise)
            points = plt.scatter(x, y, s=5, c=z, cmap=cmap)
            f.colorbar(points)
            plt.show()


def plot_latent_susceptibility(latents: dict, random_flip: bool, structure: str):
    dists = list(latents.keys())
    for dist in dists:
        noises = list(latents[dist].keys())
        latent = list(latents[dist].values())
        if len(latent[0][3].cpu()) == 1:
            if not random_flip:
                means = list(map(lambda x: torch.mean(x[2]).cpu().detach().numpy(), latent))
            else:
                means = list(map(lambda x: torch.mean(torch.abs(x[2])).cpu().detach().numpy(), latent))
            means = smooth(means, 5)
            # noises = np.array(list(map(lambda x: 4 * (1 - x) ** 3 * x + 4 * (1 - x) * x ** 3, noises)))
            noises = np.array(list(map(lambda x: 2 / (np.log((1 - x) / x)), noises)))
            # delta = noises[1] - noises[0] Attention!!! Not correct anymore with non-uniform noise array
            # der = np.zeros(len(means))
            # der[0] = (means[1] - means[0])/delta
            # for i in np.arange(1, len(means)-1):
            #     der[i] = (means[i+1] - means[i-1])/(2*delta)
            # der[-1] = (means[-1] - means[-2])/delta
            der = (np.array(list(map(lambda x: torch.mean(x[0] ** 2).cpu().detach().numpy(), latent))) -
                   np.array(list(map(lambda x: torch.mean(torch.abs(x[0])).cpu().detach().numpy() ** 2, latent))))[
                  1:]  # / noises[1:]
            # print(der)
            fig, ax1 = plt.subplots()
            ax2 = ax1.twinx()
            for j, noise in enumerate(noises):
                ax1.scatter(noise, means[j], s=2, color='black')
                if noise > 1e-5:
                    ax2.scatter(noise, der[j - 1], s=5, color='blue')
            ax1.tick_params(axis='y', labelcolor='black')
            ax1.set_xlabel('bitflip probability p')
            ax1.set_ylabel(r'mean $\langle | \mu | \rangle$')
            ax2.tick_params(axis='y', labelcolor='blue')
            ax2.set_ylabel('derivative')

            plt.vlines(0.95, 0, max(der), colors='red', linestyles='dashed')
            # ax2.set_ylim([-1, 10])
            plt.xlim(0, 2)

            def gaussian(x, A, mu, sigma):
                return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

            # popt, pcov = curve_fit(gaussian, np.array(noises)[40:160], der[40:160], p0=[-8, 0.1, 0.05])
            # print(popt)

            #ax2.plot(noises, gaussian(np.array(noises), *popt))

            if random_flip:
                plt.title(structure + " + random flip")
            else:
                plt.title(structure)
            plt.show()


def scatter_latent_var(latents: dict, random_flip: bool, structure: str):
    dists = list(latents.keys())
    for dist in dists:
        noises = list(latents[dist].keys())
        latent = list(latents[dist].values())
        noises = np.array(list(map(lambda x: 2 / (np.log((1 - x) / x)), noises)))
        for j, noise in enumerate(noises):
            for h in range(0, len(latent[j][0].cpu()), 1000):
                plt.scatter(noise, latent[j][0].cpu()[h], s=3, color='black')
        #plt.ylim([-0.00001, 0.00001])
        plt.xlabel('bitflip probability p')
        plt.ylabel(r'latent space mean $\mu$')
        # plt.xscale('log')
        plt.xlim(0, 2)
        if random_flip:
            plt.title(structure + " + random flip")
        else:
            plt.title(structure)
        plt.show()


def plot_reconstruction_error(reconstructions: dict, random_flip: bool, structure: str):
    dists = list(reconstructions.keys())
    for dist in dists:
        noises = list(reconstructions[dist].keys())
        noises = np.array(list(map(lambda x: 2 / (np.log((1 - x) / x)), noises)))
        reconstruction = list(reconstructions[dist].values())
        rec_mean = np.array(list(map(lambda x: np.array(x[0].cpu().data), reconstruction)))
        rec_var = np.array(list(map(lambda x: np.array(x[1].cpu().data), reconstruction)))
        # rec_mean = np.array(list(map(lambda x: np.array(x.cpu().data), reconstruction)))
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(noises, rec_mean)
        ax2.plot(noises, rec_var)
        ax1.set_xlabel('associated temperature')
        ax1.set_ylabel('reconstruction error')
        if random_flip:
            plt.title(structure + " + random flip")
        else:
            plt.title(structure)
        # plt.xscale('log')
        plt.show()


def plot_reconstruction_derivative(reconstructions: dict, random_flip: bool, structure: str):
    dists = list(reconstructions.keys())
    for dist in dists:
        noises = list(reconstructions[dist].keys())
        # noises = np.array(list(map(lambda x: 4 * (1 - x) ** 3 * x + 4 * (1 - x) * x ** 3, noises)))
        noises = np.array(list(map(lambda x: 2 / (np.log((1 - x) / x)), noises)))
        reconstruction = list(reconstructions[dist].values())
        reconstruction = np.array(list(map(lambda x: np.array(x[0].cpu().data), reconstruction)))
        reconstruction = smooth(reconstruction, 17)
        # TODO write as function
        # der = np.zeros(len(reconstruction))
        # der[0] = (reconstruction[1] - reconstruction[0]) / (noises[1] - noises[0])
        # for i in np.arange(1, len(reconstruction) - 1):
        #     der[i] = (reconstruction[i + 1] - reconstruction[i - 1]) / (noises[i + 1] - noises[i - 1])
        # der[-1] = (reconstruction[-1] - reconstruction[-2]) / (noises[-1] - noises[-2])
        der = np.gradient(reconstruction, noises)
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        act_noises = np.exp(-2 / noises) / (1 + np.exp(-2 / noises))
        syndrome_p = 4 * (act_noises ** 3 * (1 - act_noises) + act_noises * (1 - act_noises) ** 3)
        ax1.plot(noises, reconstruction / syndrome_p, color='black')
        ax1.set_ylim(0, 1)
        plt.vlines(0.95, -0.1, 0.1, colors='red', linestyles='dashed')
        ax2.plot(noises[8:-8], der[8:-8], color='blue')
        ax1.tick_params(axis='y', labelcolor='black')
        ax1.set_xlabel('bitflip probability p')
        ax1.set_ylabel('reconstruction error')
        ax2.tick_params(axis='y', labelcolor='blue')
        # plt.xscale('log')
        plt.xlim(0, 2)
        if random_flip:
            plt.title(structure + " + random flip")
        else:
            plt.title(structure)
        plt.show()


def plot_reconstruction(sample, noise, distance, model):
    model.load()
    model.eval()
    plt.imshow(sample.cpu().numpy()[0, 0], cmap='magma')
    print(np.sum(sample))
    print((distance ** 2 - (np.sum(sample) if np.sum(sample) > 0 else -np.sum(sample))) / (2 * distance ** 2))
    print(4 * (noise ** 3 * (1 - noise) + noise * (1 - noise) ** 3))
    plt.colorbar()
    plt.show()
    with torch.no_grad():
        output, mean, log_var = model.forward(sample.get_syndromes())
        # output = torch.where(output > 0.5, torch.ones_like(output), torch.zeros_like(output))
    plt.imshow(output.cpu().data.numpy()[0, 0], cmap='magma')
    plt.colorbar()
    plt.show()
    print(loss_func(output, mean, log_var, sample.get_syndromes()))


def plot_mean_variance_samples(raw, distance, noise_model):  # TODO delete if other function works
    results = raw.get(distance)
    noises = list(results.keys())
    # noises = np.array(list(map(lambda x: 4 * (1 - x) ** 3 * x + 4 * (1 - x) * x ** 3, noises)))
    # noises = np.array(list(map(lambda x: 2 / (np.log((1 - x) / x)), noises)))
    if noise_model == 'BitFlip':
        for noise in noises:
            # mean = torch.mean(results[noise][0])
            for i in np.arange(0, len(results[noise][0]), int(len(results[noise][0]) / 5)):
                plt.scatter(2 / (np.log((1 - noise) / noise)), results[noise][0][i].cpu().detach().numpy(), c='black',
                            s=3)
    elif noise_model == 'Depolarizing':
        for noise in noises:
            mean = torch.mean(results[noise][0], dim=1)
            for i in np.arange(0, len(mean), int(len(mean) / 5)):
                plt.scatter(2 / (np.log((1 - noise) / noise)), mean[i].cpu().detach().numpy(), c='black',
                            s=3)
    # plt.xscale('log')
    plt.show()
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    if noise_model == 'BitFlip':
        for noise in noises:
            ax1.scatter(2 / (np.log((1 - noise) / noise)),
                        torch.mean(torch.abs(results[noise][0])).cpu().detach().numpy(), c='black', s=3)
            ax2.scatter(2 / (np.log((1 - noise) / noise)),
                        (torch.mean(results[noise][0] ** 2).cpu().detach().numpy() - torch.mean(
                            torch.abs(results[noise][0])).cpu().detach().numpy() ** 2), c='blue', s=3)
    elif noise_model == 'Depolarizing':
        for noise in noises:
            mean = torch.mean(results[noise][0], dim=1)
            ax1.scatter(2 / (np.log((1 - noise) / noise)),
                        torch.mean(torch.abs(mean)).cpu().detach().numpy(), c='black', s=3)
            ax2.scatter(2 / (np.log((1 - noise) / noise)),
                        (torch.mean(mean ** 2).cpu().detach().numpy() - torch.mean(
                            torch.abs(mean)).cpu().detach().numpy() ** 2), c='blue', s=3)
            ax2.scatter(2 / (np.log((1 - noise) / noise)),
                        (torch.mean(results[noise][0][0] * results[noise][0][1])
                        - torch.mean(results[noise][0][0]) * torch.mean(results[noise][0][1])).cpu().detach().numpy(),
                        c='red', s=3)
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.set_xlabel('bitflip probability p')
    ax1.set_ylabel(r'mean $\langle | \mu | \rangle$')
    ax2.tick_params(axis='y', labelcolor='blue')
    ax2.set_ylabel('susceptibility')
    plt.tight_layout()
    if noise_model == 'BitFlip':
        plt.vlines(0.95, 0, 0.0012)
    elif noise_model == 'Depolarizing':
        plt.vlines(1.22, 0, 0.0012)
    plt.show()

def plot_loss(history, distance, val=True, save=False):
    epochs = len(history.history['accuracy'])
    # summarize history for loss
    plt.plot(np.arange(epochs), history.history['loss'])
    plt.plot(np.arange(epochs), history.history['val_loss'])
    plt.title('model loss L={0}'.format(distance))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.grid()
    if save:
        plt.savefig(str(Path().resolve().parent) + "/data/loss.pdf", bbox_inches = 'tight')
    plt.show()


def plot_collapsed(dictionary, noises, pc, nu):
    """

    :param nu:
    :param pc:
    :param noises:
    :param dictionary: dictionary: keys: distances, labels: tuple (noises, predictions)
    :return:
    """
    dists_str = dictionary.keys()
    dists = np.array(list(dists_str)).astype(int)
    markers = ['o', 'x', 's', 'v', '+', '^']
    colors = ['blue', 'black', 'green', 'red', 'orange', 'purple']
    for i, dist in enumerate(dists):
        predictions, errors = dictionary[str(dist)]
        noises_temp = np.array(list(map(lambda x: (x - pc) * dist ** (1 / nu), noises)))
        plt.plot(noises_temp, predictions[:, 0], color=colors[i], marker=markers[i], linewidth=1, label=dist)
        plt.plot(noises_temp, predictions[:, 1], color=colors[i], marker=markers[i], linewidth=1, label=dist)
    plt.legend()
    plt.xlabel(r'noise probability $p$')
    plt.ylabel(r'ordered/unordered')
    plt.show()
