import tensorflow as tf
import torch
from abc import ABC, abstractmethod
from ErrorCode import BitFlipSurface, DepolarizingSurface
import numpy as np


# TODO function that creates data for an array of different distances


# TODO: Dataset als abstract class, for different code types inherit of dataset abstract class
class Dataset(ABC):
    def __init__(self, distance, noises, name):
        self.syndromes = None
        self.labels = None
        self.distance = distance
        self.noises = noises
        if not (type(name) is str):
            raise ValueError
        self.name = name

    def save(self):
        tf.data.Dataset.save(self.syndromes, "syndromes_{0}.dat".format(self.name))
        tf.data.Dataset.save(self.labels, "labels_{0}.dat".format(self.name))

    def load(self):
        self.syndromes = tf.data.Dataset.load("syndromes_{0}.dat".format(self.name))
        self.labels = tf.data.Dataset.load("labels_{0}.dat".format(self.name))

    def get_syndromes(self, batch_size):
        return self.syndromes.batch(batch_size)

    @abstractmethod
    def generate_data(self, n, rounds):
        raise NotImplementedError

    @abstractmethod
    def prepare_data(self, layout):
        raise NotImplementedError


class BitFlipSurfaceData(Dataset):
    def __init__(self, distance, noises, name):
        super().__init__(distance, noises, name)

    def generate_data(self, n, rounds):
        # Important: generate data alternating between noise, otherwise data will be ordered as labels = [0, ..., 0, 1, ..., 1]
        # Shuffling with limited buffer_size then will not mix 0 and 1 labels, shuffling with buffer over whole data set negates the effect of memory saving since the memory occupancy will be too large for large datasets
        for i in np.arange(rounds):
            for noise in self.noises:
                code = BitFlipSurface(self.distance, noise)
                syndromes, labels = code.get_syndromes(int(n / rounds))
                syndromes = tf.data.Dataset.from_tensor_slices(syndromes)
                # Maps boolean True/False values to -1/1 integers as we have them in our Ising model description
                syndromes = syndromes.map(lambda x: tf.where(x, -1, 1))
                labels = tf.data.Dataset.from_tensor_slices(labels)
                if self.syndromes is None:
                    self.syndromes = syndromes
                    self.labels = labels
                else:
                    self.syndromes = self.syndromes.concatenate(syndromes)
                    self.labels = self.labels.concatenate(labels)
        return self

    def prepare_data(self, layout):
        def ising_shape(arr):
            result = np.zeros((self.distance - 1, self.distance, 1))
            shift = -1
            for k in np.arange(self.distance + 1):
                if k % 2 == 0:
                    shift += 1
                for j in np.arange(int((self.distance - 1) / 2)):
                    result[j + shift, -int((self.distance - 1) / 2) + j - k + shift] = arr[
                        k * int((self.distance - 1) / 2) + j]
            return result

        if layout == 0:
            pass
        if layout == 1 or layout == 2:
            self.syndromes = self.syndromes.map(
                lambda x: tf.reshape(x, [self.distance + 1, int(np.round((self.distance - 1) / 2))]))
        if layout == 3:
            syndromes_new = np.zeros((self.syndromes.__len__(), self.distance - 1, self.distance, 1))
            for i, elm in enumerate(self.syndromes):
                elm_list = tf.get_static_value(elm)
                syndromes_new[i] = ising_shape(elm_list)
            self.syndromes = tf.data.Dataset.from_tensor_slices(syndromes_new)
        return self

    def get_training_data(self, ratio, batch_size):
        dataset = tf.data.Dataset.zip((self.syndromes, self.labels))
        num = dataset.__len__().numpy()
        dataset = dataset.shuffle(buffer_size=num)
        dataset_train = dataset.take(int(ratio * num))
        dataset_val = dataset.skip(int(ratio * num))
        dataset_train = dataset_train.batch(batch_size)
        # Also batch the validation data, otherwise we run into errors in the shape of the data when training the model
        dataset_val = dataset_val.batch(batch_size)
        return dataset_train, dataset_val


class DepolarizingSurfaceData(Dataset):
    def __init__(self, distance, noises, name):
        super().__init__(distance, noises, name)

    def generate_data(self, n, rounds):
        # Important: generate data alternating between noise, otherwise data will be ordered as labels = [0, ..., 0,
        # 1, ..., 1] Shuffling with limited buffer_size then will not mix 0 and 1 labels, shuffling with buffer over
        # whole data set negates the effect of memory saving since the memory occupancy will be too large for large
        # datasets
        for i in np.arange(rounds):
            for noise in self.noises:
                code = DepolarizingSurface(self.distance, noise)
                syndromes, labels = code.get_syndromes(int(n / rounds))
                syndromes = tf.data.Dataset.from_tensor_slices(syndromes)
                # Maps boolean True/False values to -1/1 integers as we have them in our Ising model description
                syndromes = syndromes.map(lambda x: tf.where(x, -1, 1))
                labels = tf.data.Dataset.from_tensor_slices(labels)
                if self.syndromes is None:
                    self.syndromes = syndromes
                    self.labels = labels
                else:
                    self.syndromes = self.syndromes.concatenate(syndromes)
                    self.labels = self.labels.concatenate(labels)
        return self

    def prepare_data(self, layout):
        def order_shape(arr):
            result = np.zeros((self.distance + 1, self.distance + 1, 3))
            z_stabs = np.zeros((self.distance + 1, self.distance + 1))
            x_stabs = np.zeros((self.distance + 1, self.distance + 1))
            all_stabs = np.zeros((self.distance + 1, self.distance + 1))
            shift = True
            for k in np.arange(self.distance + 1):
                for h in np.arange(1, self.distance, 2):
                    if shift:
                        z_stabs[k, h + 1] = arr[k * int((self.distance - 1) / 2) + int((h - 1) / 2)] # h starts at 1,
                        # with 2 increment!
                        all_stabs[k, h + 1] = arr[k * int((self.distance - 1) / 2) + int((h - 1) / 2)]
                    elif not shift:
                        z_stabs[k, h] = arr[k * int((self.distance - 1) / 2) + int((h - 1) / 2)]
                        all_stabs[k, h] = arr[k * int((self.distance - 1) / 2) + int((h - 1) / 2)]
                shift = not shift
            shift = False
            for k in np.arange(1, self.distance):
                for h in np.arange(0, self.distance + 1, 2):
                    if shift:
                        x_stabs[k, h + 1] = arr[
                            (k - 1) * int((self.distance + 1) / 2) + int(h / 2) + int((self.distance ** 2 - 1) / 2)]
                        all_stabs[k, h + 1] = arr[
                            (k - 1) * int((self.distance + 1) / 2) + int(h / 2) + int((self.distance ** 2 - 1) / 2)]
                    elif not shift:
                        x_stabs[k, h] = arr[(k - 1) * int((self.distance + 1) / 2) + int(h / 2) + int((self.distance ** 2 - 1) / 2)]
                        all_stabs[k, h] = arr[(k - 1) * int((self.distance + 1) / 2) + int(h / 2) + int((self.distance ** 2 - 1) / 2)]
                shift = not shift
            result[:, :, 0] = all_stabs
            result[:, :, 1] = z_stabs
            result[:, :, 2] = x_stabs
            return result

        if layout == 0:
            pass
        if layout == 1 or layout == 2:
            pass
        if layout == 3:
            syndromes_new = np.zeros((self.syndromes.__len__(), self.distance + 1, self.distance + 1, 3))
            # 1: all stabilizers, 2: only Z stabilizers, 3: only X stabilizers
            for i, elm in enumerate(self.syndromes):
                elm_list = tf.get_static_value(elm)
                syndromes_new[i] = order_shape(elm_list)
            self.syndromes = tf.data.Dataset.from_tensor_slices(syndromes_new)
        return self

    def get_training_data(self, ratio, batch_size):
        dataset = tf.data.Dataset.zip((self.syndromes, self.labels))
        num = dataset.__len__().numpy()
        dataset = dataset.shuffle(buffer_size=num)
        dataset_train = dataset.take(int(ratio * num))
        dataset_val = dataset.skip(int(ratio * num))
        dataset_train = dataset_train.batch(batch_size)
        # Also batch the validation data, otherwise we run into errors in the shape of the data when training the model
        dataset_val = dataset_val.batch(batch_size)
        return dataset_train, dataset_val
