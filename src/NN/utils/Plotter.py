import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np


class Plotter:

    @classmethod
    def plot_history(cls, history, val=True):
        epochs = len(history.history['loss'])
        fig = plt.figure()
        # set height ratios for subplots
        gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])

        # the first subplot
        ax0 = plt.subplot(gs[0])
        # log scale for axis Y of the first subplot
        ax0.set_yscale("linear")
        ax0.grid()
        line0, = ax0.plot(epochs, history.history['accuracy'], color='k', linestyle='solid')
        if val:
            line02, = ax0.plot(epochs, history.history['val_accuracy'], color='k', linestyle='dashed')

        # the second subplot
        # shared axis X
        ax1 = plt.subplot(gs[1], sharex=ax0)
        line1, = ax1.plot(epochs, history.history['loss'], color='k', linestyle='solid')
        if val:
            line12, = ax1.plot(epochs, history.history['val_loss'], color='k', linestyle='dashed')

        plt.setp(ax0.get_xticklabels(), visible=False)
        ax1.grid()

        yticks = ax1.yaxis.get_major_ticks()
        yticks[-1].label1.set_visible(False)

        # put legend on first subplot
        ax0.legend((line0, line02), ('accuracy', 'val_accuracy'), loc='lower right')
        ax1.legend((line1, line12), ('loss', 'val_loss'), loc='lower right')
        plt.subplots_adjust(hspace=.0)

        plt.show()

    @classmethod
    def plot_accuracy(cls, history, distance, val=True):
        epochs = len(history.history['accuracy'])
        plt.plot(np.arange(epochs), history.history['accuracy'])
        if val:
            plt.plot(np.arange(epochs), history.history['val_accuracy'])
        plt.title('model accuracy distance={0}'.format(distance))
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.grid()
        # plt.savefig(r'C:\Users\Dominik\Documents\Uni-Dokumente\HiWi FZJ\Dok\Pics\DNN_acc_L=32.pdf', bbox_inches = 'tight')
        plt.show()

    @classmethod
    def plot_loss(cls, history, distance, val=True):
        epochs = len(history.history['accuracy'])
        # summarize history for loss
        plt.plot(np.arange(epochs), history.history['loss'])
        plt.plot(np.arange(epochs), history.history['val_loss'])
        plt.title('model loss L={0}'.format(distance))
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.grid()
        # plt.savefig(r'C:\Users\Dominik\Documents\Uni-Dokumente\HiWi FZJ\Dok\Pics\DNN_loss_L=32.pdf', bbox_inches = 'tight')
        plt.show()

    @classmethod
    def plot_prediction(cls, dictionary, noises):
        """

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
            errors = errors/1
            plt.errorbar(noises, predictions[:, 0], yerr = errors[:,0], color=colors[i], marker=markers[i], linewidth=1, label=dist)
            plt.errorbar(noises, predictions[:, 1], yerr = errors[:,1], color=colors[i], marker=markers[i], linewidth=1, label=dist)
        plt.legend()
        plt.xlabel(r'noise probability $p$')
        plt.ylabel(r'ordered/unordered')
        plt.show()

    @classmethod
    def plot_collapsed(cls, dictionary, noises, pc, nu):
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
            noises_temp = np.array(list(map(lambda x: (x-pc)*dist**(1/nu), noises)))
            plt.plot(noises_temp, predictions[:, 0], color=colors[i], marker=markers[i], linewidth=1, label=dist)
            plt.plot(noises_temp, predictions[:, 1], color=colors[i], marker=markers[i], linewidth=1, label=dist)
        plt.legend()
        plt.xlabel(r'noise probability $p$')
        plt.ylabel(r'ordered/unordered')
        plt.show()
