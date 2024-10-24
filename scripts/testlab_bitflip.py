import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.nn.data.bitflip import BitFlipToricData
from src.nn.utils.functions import smooth


def p_nts(noise):
    return 4 * ((1 - noise) * noise ** 3 + (1 - noise) ** 3 * noise)


def p_nn(noise):
    return 6 * noise * (1 - noise) ** 6 + 20 * noise ** 3 * (1 - noise) ** 4 + 6 * noise ** 5 * (
            1 - noise) ** 2 + 6 * noise ** 2 * (1 - noise) ** 5 + 20 * noise ** 4 * (
            1 - noise) ** 3 + 6 * noise ** 6 * (1 - noise)


def p_o(noise):
    return 8 * noise * (1 - noise) ** 7 + 56 * noise ** 3 * (1 - noise) ** 5 + 56 * noise ** 5 * (
            1 - noise) ** 3 + 8 * noise ** 7 * (1 - noise)


def var(noise):
    #return 1 / distance ** 2 * (1 + 4 * (1 - 2 * p_nn(noise)) + (distance ** 2 - 5) * (1 - 2 * p_o(noise))) - (
    #            1 - 2 * p_nts(noise)) ** 2
    return 1 / distance ** 2 * (1 + 4 * (1 - 2 * p_nn(noise)) + (distance ** 2 - 5) * (1 - 2 * p_nts(noise)) ** 2) - (
            1 - 2 * p_nts(noise)) ** 2


def z(p):
    pass


def create_configs(sz):
    pass


def hamiltonian(p, bonds, config):
    def pb(int):
        max = np.shape(bonds)[0]
        if int == max:
            return 0
        elif int == -1:
            return max - 1
        else:
            return int

    beta_J = 1 / 2 * np.log((1 - p) / p)
    n, m = np.shape(bonds)
    h = 0
    for i in range(n):
        for j in range(m):
            h += bonds[i, j] * config[i, j] * config[pb(i + 1), j]
            h += bonds[i, j] * config[i, j] * config[pb(i - 1), j]
            h += bonds[i, j] * config[i, j] * config[i, pb(j + 1)]
            h += bonds[i, j] * config[i, j] * config[i, pb(j - 1)]
    return -1 * beta_J * h


def sus(noise):
    squared = 1 / distance ** 2 * (1 + 4 * (1 - 2 * p_nn(noise)) + (distance ** 2 - 5) * (1 - 2 * p_nts(noise)) ** 2)
    abs_s = 0
    return squared - abs_s ** 2


distance = 21
task = 3

if task == 0:
    # temperature = 1
    # noise = np.exp(-2 / temperature) / (1 + np.exp(-2 / temperature))
    noise = 0.109
    n = 100
    p = 4 * ((1 - noise) * noise ** 3 + (1 - noise) ** 3 * noise)
    pnn = 6 * noise * (1 - noise) ** 6 + 20 * noise ** 3 * (1 - noise) ** 4 + 6 * noise ** 5 * (1 - noise) ** 2
    po = 8 * noise * (1 - noise) ** 7 + 56 * noise ** 3 * (1 - noise) ** 5 + 56 * noise ** 5 * (
            1 - noise) ** 3 + 8 * noise ** 7 * (1 - noise)
    sample = BitFlipToricData(distance=distance, noises=[noise],
                              name="BFS_Testing-{0}".format(distance),
                              load=False, random_flip=False, sequential=False).training().initialize(n)
    mean = torch.mean(sample.syndromes, dim=(1, 2, 3))
    s = sample.syndromes.cpu().detach().numpy()
    # print(mean)
    # print(torch.as_tensor(np.random.normal(1 - 2 * p, 1 - (1 - 2 * p) ** 2, 100)))
    print('Variance:', torch.var(mean))

    temperatures = np.arange(0, 2, 0.01)
    noises = np.exp(-2 / temperatures) / (1 + np.exp(-2 / temperatures))

    plt.plot(temperatures, 1 - 2 * p_nts(noises))
    plt.plot(temperatures, 30 * np.gradient(p_nts(noises)))
    plt.vlines(0.951, 0, 0.5)
    plt.show()
elif task == -1:
    def two_body_exp(samples):
        prods = np.array(list(map(lambda x: x[0, 0, 0] * x[0, 0, 2], samples)))
        return np.mean(prods)


    noise = 0.1
    n = 10000
    sample = BitFlipToricData(distance=distance, noises=[noise],
                              name="BFS_Testing-{0}".format(distance),
                              load=False, random_flip=False, sequential=False).training().initialize(n)
    s = sample.syndromes.cpu().detach().numpy()
    po = 8 * noise * (1 - noise) ** 7 + 56 * noise ** 3 * (1 - noise) ** 5 + 56 * noise ** 5 * (
            1 - noise) ** 3 + 8 * noise ** 7 * (1 - noise)
    print(two_body_exp(s))
    print(1 - 2 * po)
    print((1 - 2 * p_nts(noise)) ** 2)
    # print(np.abs((two_body_exp(s) - (1 - 2 * po))/two_body_exp(s)))

elif task == 1:
    n1 = 1000
    temperature = 1
    # temperature = np.array([])
    noise = np.exp(-2 / temperature) / (1 + np.exp(-2 / temperature))
    # print(noises)
    distances = np.array([5, 11, 17, 25, 31, 33, 39, 45, 51, 59, 63])
    for dist in tqdm(distances):
        p = 4 * ((1 - noise) * noise ** 3 + (1 - noise) ** 3 * noise)
        sample = BitFlipToricData(distance=dist, noises=[noise],
                                  name="BFS_Testing-{0}".format(dist),
                                  load=False, random_flip=True, sequential=False).training().initialize(n1)
        mean = torch.mean(sample.syndromes, dim=(1, 2, 3))
        # print(mean)
        # print(torch.as_tensor(np.random.normal(1 - 2 * p, 1 - (1 - 2 * p) ** 2, 100)))
        # print(s)
        plt.scatter(1 / dist, torch.mean(torch.abs(mean)).cpu(), color='red')
        # print('Need to obtain:', torch.var(mean))
        # print(s.shape)

    # ps = np.array(list(map(lambda x: np.exp(-2 / x) / (1 + np.exp(-2 / x)), Ts)))

    # plt.plot(Ts, var(ps), label='variance m')
    # plt.plot(Ts, sus(ps), label='susceptibility m')
    # plt.vlines(0.95, 0., 1., colors='red', linestyles='dashed')
    # plt.legend()
    plt.show()
elif task == 2:
    n1 = 1000
    distances = np.array([5, 11, 17, 25, 31])
    temperature = np.arange(0.1, 2.7, 0.1)
    # temperature = np.array([])
    noises = np.exp(-2 / temperature) / (1 + np.exp(-2 / temperature))
    # print(noises)
    coloring = ['red', 'green', 'blue', 'yellow', 'black', 'purple']
    for k, dist in enumerate(distances):
        print('Distance: ', dist)
        binder = np.zeros(len(noises))
        print('Iterations: ', len(noises))
        for i, noise in tqdm(enumerate(noises)):
            sample = BitFlipToricData(distance=dist, noises=[noise],
                                      name="BFS_Testing-{0}".format(dist),
                                      load=False, random_flip=True, sequential=False).training().initialize(n1)
            mean = torch.mean(sample.syndromes, dim=(1, 2, 3))
            # plt.scatter(2 / np.log((1 - noise) / noise), torch.mean(torch.abs(mean)).cpu(), color='red')
            binder[i] = 1 - (torch.mean(mean ** 4) / (3 * torch.mean(mean ** 2) ** 2)).cpu()
            # print('Need to obtain:', torch.var(mean))
            # print(s.shape)
        # plt.plot(2 / np.log((1 - noises) / noises), binder, color=coloring[k])
        plt.plot(noises, binder, color=coloring[k])

    Ts = temperature
    ps = np.array(list(map(lambda x: np.exp(-2 / x) / (1 + np.exp(-2 / x)), Ts)))

    # plt.plot(Ts, var(ps), label='variance m')
    # plt.plot(Ts, sus(ps), label='susceptibility m')
    # plt.vlines(0.95, 0., 1., colors='red', linestyles='dashed')
    plt.vlines(0.109, 0, 1, color='red', linestyles='dashed')
    plt.xlim(0.05, 0.15)
    plt.legend()
    plt.show()
elif task == 20:  # with 1, 0 instead of -1, 1
    n1 = 1000000
    temperature = np.arange(0.90, 1.1, 0.01)
    noises = np.exp(-2 / temperature) / (1 + np.exp(-2 / temperature))
    # print(noises)
    for noise in tqdm(noises):
        p = 4 * ((1 - noise) * noise ** 3 + (1 - noise) ** 3 * noise)
        sample = BitFlipToricData(distance=distance, noises=[noise],
                                  name="BFS_Testing-{0}".format(distance),
                                  load=False, random_flip=False, sequential=False).training().initialize(n1)
        nd_sample = sample.syndromes.cpu().detach().numpy()
        shape = np.shape(nd_sample)
        nd_sample = nd_sample.flatten()
        nd_sample = np.array(list(map(lambda x: 0 if x == 1 else 1, nd_sample)))
        nd_sample = np.reshape(nd_sample, shape)
        sample = torch.as_tensor(nd_sample, dtype=torch.float)
        mean = torch.mean(sample, dim=(1, 2, 3))
        # mean = torch.mean(sample.syndromes, dim=(1, 2, 3))
        # print(mean)
        # print(torch.as_tensor(np.random.normal(1 - 2 * p, 1 - (1 - 2 * p) ** 2, 100)))
        # print(s)
        plt.scatter(2 / np.log((1 - noise) / noise), torch.var(mean).cpu(), color='red')
        plt.scatter(2 / np.log((1 - noise) / noise),
                    (torch.mean(mean ** 2) - torch.mean(torch.abs(mean)) ** 2).cpu().detach().numpy(), color='blue')
        # print('Need to obtain:', torch.var(mean))
        # print(s.shape)

    Ts = np.arange(0, 3, 0.01)
    ps = np.array(list(map(lambda x: np.exp(-2 / x) / (1 + np.exp(-2 / x)), Ts)))

    # plt.plot(Ts, var(ps), label='variance m')
    plt.legend()
    plt.show()
elif task == 3:
    n1 = 50000
    temperature = np.arange(0.02, 3, 0.02)
    noises = np.exp(-2 / temperature) / (1 + np.exp(-2 / temperature))
    sus = np.zeros(len(noises))
    means = np.zeros(len(noises))
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    for i, noise in enumerate(tqdm(noises)):
        p = 4 * ((1 - noise) * noise ** 3 + (1 - noise) ** 3 * noise)
        sample = BitFlipToricData(distance=distance, noises=[noise],
                                  name="BFS_Testing-{0}".format(distance),
                                  load=False, random_flip=True, sequential=False,
                                  device=torch.device('cpu')).training().initialize(n1)
        mean = torch.mean(sample.syndromes, dim=(1, 2, 3))
        # mean = torch.log(mean)
        # print(noise)
        # print(mean)
        sus[i] = (torch.mean(mean ** 2) - torch.mean(torch.abs(mean)) ** 2).cpu().detach().numpy()
        means[i] = torch.mean(torch.abs(mean)).cpu().detach().numpy()
        # print('Need to obtain:', torch.var(mean))
        # print(s.shape)


    ax2.plot(2 / np.log((1 - noises) / noises), smooth(sus, 5), color='blue', label='susceptibility')
    plt.vlines(0.95, 0, max(sus), colors='red', linestyles='dashed', label='threshold')
    ps = np.array(list(map(lambda x: np.exp(-2 / x) / (1 + np.exp(-2 / x)), temperature)))
    print(noises[np.argmax(sus)])
    ax1.plot(temperature, 1 - 2 * 4 * ((1 - noises) * noises ** 3 + (1 - noises) ** 3 * noises), color='black', label='number of NTS')
    plt.legend()
    ax1.set_xlabel('associated temperature')
    ax1.set_ylabel(r'mean $\langle M \rangle$')
    ax2.set_ylabel(r'$d \cdot$ susceptibility')
    plt.show()

    np.save('mean.npy', means)
    np.save('sus.npy', sus)
    print(sus)
