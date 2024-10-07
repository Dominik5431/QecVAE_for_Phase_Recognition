import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.nn.data.depolarizing import DepolarizingToricData, DepolarizingSurfaceData
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


def var_bf(noise):
    #return 1 / distance ** 2 * (1 + 4 * (1 - 2 * p_nn(noise)) + (distance ** 2 - 5) * (1 - 2 * p_o(noise))) - (
    #            1 - 2 * p_nts(noise)) ** 2
    return 1 / distance ** 2 * (1 + 4 * (1 - 2 * p_nn(noise)) + (distance ** 2 - 5) * (1 - 2 * p_nts(noise)) ** 2) - (
            1 - 2 * p_nts(noise)) ** 2
    # return (1 / (distance ** 2 - 1) ** 2 * (
    #        distance ** 2 * (1 + 4 * (1 - 2 * p_nn(noise)) + (distance ** 2 - 5) * (1 - 2 * p_nts(noise)) ** 2)
    #        - 2 * (1 + 4 * (1 - 2 * p_nn(noise)) + (distance ** 2 - 5) * (1 - 2 * p_nts(noise)) ** 2)
    #        + 1)
    #        - (1 - 2 * p_nts(noise)) ** 2)  # here without last element --> exact same result as when included


def p_nn_xz(noise):
    return 2 * p_nts(noise / 3) * (1 - p_nts(noise / 3)) + 4 * (noise / 3) * (1 - noise / 3) ** 5 + 8 * (
            noise / 3) ** 2 * (1 - noise / 3) ** 4 + 4 * (noise / 3) ** 3 * (1 - noise / 3) ** 3 + 8 * (
            noise / 3) ** 4 * (1 - noise / 3) ** 2


def correlation(noise):
    return 1 / distance ** 2 * (
            4 * (1 - 2 * p_nn_xz(noise)) + (distance ** 2 - 4) * (1 - 2 * p_nts(2 / 3 * noise)) ** 2) - (
            1 - 2 * p_nts(2 / 3 * noise)) ** 2


distance = 5
distances = [3, 5, 7, 9, 11, 17, 27] # , 37]
task = 2

if task == 0:
    n1 = 1000
    # temperature = np.arange(0.8, 1.5, 0.05)
    temperature = np.arange(0.2, 2.6, 0.2)
    # temperature = np.array([])
    noises = np.exp(-4 / temperature) / (1 / 3 + np.exp(-4 / temperature))
    # print(noises)
    for noise in tqdm(noises):
        p = p_nts(noise)
        sample = DepolarizingToricData(distance=distance, noises=[noise],
                                       name="BFS_Testing-{0}".format(distance),
                                       load=False, random_flip=False, sequential=False).training().initialize(n1)
        mean = torch.mean(sample.syndromes, dim=(2, 3))
        mx = mean[:, 0]
        mz = mean[:, 1]
        # plt.scatter(2 / np.log((1 - noise) / noise), var_x.cpu(), color='red')
        # plt.scatter(2 / np.log((1 - noise) / noise), var_z.cpu(), color='blue')
        plt.scatter(4 / np.log(3 * (1 - noise) / noise), torch.mean(mx).cpu(),
                    color='black')  # this is just mean of blue and red
        plt.scatter(4 / np.log(3 * (1 - noise) / noise), torch.mean(mz).cpu(), color='green')
        plt.scatter(4 / np.log(3 * (1 - noise) / noise), torch.mean(mean).cpu(), color='orange')
        # plt.scatter(2 / np.log((1 - noise) / noise), torch.var(torch.mean(mean, dim=1)).cpu(), color='blue')
        # print('Need to obtain:', torch.var(mean))
        # print(s.shape)

    Ts = np.arange(0, 3, 0.01)
    ps = np.array(list(map(lambda x: np.exp(-4 / x) / (1 / 3 + np.exp(-4 / x)), Ts)))
    plt.vlines(1.565, 0, 0.5, colors='red', linestyles='dashed')
    plt.vlines(1.465, 0, 0.5, colors='red', linestyles='dashed')
    plt.vlines(1.250, 0, 0.5, colors='grey', linestyles='dashed')
    plt.plot(Ts, 1 - 2 * p_nts(ps), label='variance m')
    plt.plot(Ts, 1 - 2 * p_nts(2 / 3 * ps), label='variance m')
    # plt.plot(Ts, 1 / 3 * var_bf(1 / 3 * ps))
    # plt.plot(Ts, correlation(ps))
    # plt.plot(Ts, 2 * var_bf(2 / 3 * ps) + var_bf(1 / 3 * ps), color='black')
    plt.legend()
    # plt.xlim(0.9, 1.6)
    # plt.ylim(0., 0.002)
    plt.show()

    '''sample = DepolarizingSurfaceData(distance=distance, noises=[noise],
                                     name="BFS_Testing-{0}".format(distance),
                                     load=False, random_flip=False, sequential=False).training().initialize(n)
    plt.imshow(np.reshape(sample[0, 0].cpu().numpy(), (distance, distance)), cmap='magma')
    plt.colorbar()
    plt.show()
    plt.imshow(np.reshape(sample[0, 1].cpu().numpy(), (distance, distance)), cmap='magma')
    plt.colorbar()
    plt.show()'''
elif task == -1:
    p = 0.08
    print((1 - p) ** 18)
elif task == 1:
    Ts = np.arange(0, 2, 0.01)
    ps = np.array(list(map(lambda x: np.exp(-2 / x) / (1 + np.exp(-2 / x)), Ts)))
    plt.plot(Ts, 1 - 2 * p_nts(ps), label='probability')
    # plt.plot(Ts, - np.gradient(1-2*ps, Ts), label='derivative')
    plt.plot(Ts, 1 - (1 - 2 * p_nts(ps)) ** 2, label='variance')
    plt.plot(Ts, var_bf(ps) / max(var_bf(ps)), label='variance m')
    plt.vlines(0.95, 0, 1, colors='red', linestyles='dashed')
    plt.xlabel('associated temperature')
    plt.ylabel('probability of a syndrome being non-trivial')
    plt.legend()
    plt.show()
elif task == 2:
    n1 = 20000
    # temperature = np.arange(0.8, 1.5, 0.05)
    temperature = np.arange(0.1, 3., 0.1)
    # temperature = np.array([])
    noises = np.exp(-4 / temperature) / (1 / 3 + np.exp(-4 / temperature))
    sus = np.zeros(len(noises))
    # fig, ax1 = plt.subplots()
    # ax2 = ax1.twinx()
    for d in distances:
        for i, noise in enumerate(tqdm(noises)):
            p = p_nts(noise)
            sample = DepolarizingSurfaceData(distance=d, noises=[noise],
                                             name="DS_Testing-{0}".format(d),
                                             load=False, device=torch.device('mps')).training().initialize(n1)
            logical = sample.syndromes[:, -2 * d:]
            lo = torch.zeros(logical.size(0), d)
            for j in range(d):
                lo[:, j] = 1 - (1 - logical[:, 0 + j]) * (1 - logical[:, d + j])
            mean = torch.mean(lo.float())

            # mx = mean[:, 0]
            # mz = mean[:, 1]
            # print(mean)
            # print(torch.as_tensor(np.random.normal(1 - 2 * p, 1 - (1 - 2 * p) ** 2, 100)))
            # print(s)
            # corr = torch.mean(torch.abs(mx) * torch.abs(mz)) - torch.mean(torch.abs(mz)) * torch.mean(torch.abs(mx))
            # corr = torch.mean(mx * mz) - torch.mean(mz) * torch.mean(mx)
            # var_x = torch.var(mx)
            # var_z = torch.var(mz)
            # total = corr + var_x + var_z
            # sus[i] = (torch.mean(mean ** 2) - torch.mean(torch.abs(mean)) ** 2).cpu().detach().numpy()
            sus[i] = mean
            # plt.scatter(4 / np.log(3 * (1 - noise) / noise), var_x.cpu(), color='red')
            # plt.scatter(4 / np.log(3 * (1 - noise) / noise), var_z.cpu(), color='blue')
            # plt.scatter(4 / np.log(3 * (1 - noise) / noise), torch.var(mean).cpu(),
            #   color='black')  # this is just mean of blue and red
            # plt.scatter(4 / np.log(3 * (1 - noise) / noise), corr.cpu(), color='green')
            # plt.scatter(4 / np.log(3 * (1 - noise) / noise), total.cpu(), color='orange')
            # plt.scatter(4 / np.log(3 * (1 - noise) / noise), torch.var(torch.mean(mean, dim=1)).cpu(), color='blue')
            # print('Need to obtain:', torch.var(mean))
            # print(s.shape)

        # plt.plot(4 / np.log(3 * (1 - noises) / noises), 1 - sus, label='d = ' + str(d))
        plt.plot(noises, 1 - sus, label='d = ' + str(d))
        idx = np.where(noises <= 1/d)[0][-1]
        n = noises[idx]
        print(n)
        plt.scatter(n, sus[idx], color='black', linewidths=2)
        ps = np.array(list(map(lambda x: np.exp(-4 / x) / (1 / 3 + np.exp(-4 / x)), temperature)))

    # ax1.plot(temperature,
    #          1 - 2 * 4 * ((1 - 2 / 3 * noises) * 2 / 3 * noises ** 3 + (1 - 2 / 3 * noises) ** 3 * 2 / 3 * noises),
    #          color='black',
    #          label='number of NTS')
    # plt.vlines(1.565, 0.25, 1, colors='red', linestyles='dashed', label='threshold')
    # plt.hlines(0.25, 0, 3, colors='black', linestyles='dashed', label='min')
    plt.vlines(0.189, 0.25, 1, colors='red', linestyles='dashed', label='threshold')
    plt.hlines(0.25, 0, 0.4, colors='black', linestyles='dashed', label='min')
    # plt.vlines(1.465, 0, 0.5, colors='red', linestyles='dashed')
    # plt.vlines(1.250, 0, 0.5, colors='grey', linestyles='dashed')
    # plt.plot(Ts, var_bf(2 / 3 * ps), label='variance m')
    # plt.plot(Ts, 1 / 3 * var_bf(1 / 3 * ps))
    # plt.plot(Ts, correlation(ps))
    # plt.plot(Ts, 2 * var_bf(2 / 3 * ps) + var_bf(1 / 3 * ps), color='black')
    # plt.legend()
    # plt.xlim(0.9, 1.6)
    # plt.ylim(0., 0.002)
    # plt.show()
    plt.ylim(0, 1)
    plt.legend()
    plt.show()

    '''Ts = np.arange(0.1, 3, 0.001)
    ps = np.array(list(map(lambda x: np.exp(-2 / x) / (1 + np.exp(-2 / x)), Ts)))
    # f = 2 * var_bf(2/3 * ps) + 1/3 * var_bf(1/3 * ps)
    f = var_bf(ps)
    df = np.gradient(f, Ts)
    plt.plot(Ts, f)
    plt.plot(Ts, np.gradient(f, Ts))
    plt.hlines(0, 0, 3)
    print(np.where(df < 0))
    print(0.5 * (ps[np.where(df < 0)[0][0]] + ps[np.where(df < 0)[0][0] - 1]))
    print(ps[np.argmax(f)])
    plt.show()'''
elif task == 3:
    n1 = 10000
    temperature = np.arange(0.3, 3.0, 0.1)
    noises = np.exp(-2 / temperature) / (1 + np.exp(-2 / temperature))
    for noise in tqdm(noises):
        p = 4 * ((1 - noise) * noise ** 3 + (1 - noise) ** 3 * noise)
        sample = BitFlipSurfaceData(distance=distance, noises=[noise],
                                    name="BFS_Testing-{0}".format(distance),
                                    load=False, random_flip=True, sequential=False).training().initialize(n1)
        mean = torch.mean(sample.syndromes, dim=(1, 2, 3))
        sus = (torch.mean(mean ** 2) - torch.mean(torch.abs(mean)) ** 2).cpu().detach().numpy()
        plt.scatter(2 / np.log((1 - noise) / noise), sus, color='red')
        # print('Need to obtain:', torch.var(mean))
        # print(s.shape)

    Ts = np.arange(0, 3, 0.01)
    ps = np.array(list(map(lambda x: np.exp(-2 / x) / (1 + np.exp(-2 / x)), Ts)))

    plt.plot(Ts, var_bf(ps), label='variance m')
    plt.legend()
    plt.show()
