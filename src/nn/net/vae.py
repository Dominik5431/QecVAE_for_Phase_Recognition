import torch.cuda

from .net import *
from .encoder import *
from .decoder import *
import subprocess as sp
import os
import time


def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values


class VariationalAutoencoder(Net):
    def __init__(self, latent_dims: int, distance: int, name: str, structure: str, noise: str, cluster: bool = False):
        super(VariationalAutoencoder, self).__init__(name, cluster)
        channels = -1
        if noise == 'BitFlip':
            channels = 1
        elif noise == 'Ising':
            channels = 1
        elif noise == 'Depolarizing':
            channels = 2
        else:
            raise ValueError('Invalid noise for this autoencoder structure.')
        assert channels > 0
        self.structure = structure
        if structure == 'standard':
            self.encoder = VariationalEncoder(latent_dims, distance, channels)
            self.decoder = Decoder(latent_dims, distance, channels)
        elif structure == 'simple':
            self.encoder = VariationalEncoderSimple(latent_dims, distance, channels)
            self.decoder = DecoderSimple(latent_dims, distance, channels)
        elif structure == 'skip':
            self.encoder = VariationalEncoder(latent_dims, distance, channels)
            self.decoder = DecoderSkip(latent_dims, distance, channels)
        elif structure == 'deep-skip':
            self.encoder = VariationalEncoderDeep(latent_dims, distance)
            self.decoder = DecoderDeepSkip(latent_dims, distance)
        elif structure == 'deep':
            self.encoder = VariationalEncoderDeep(latent_dims, distance)
            self.decoder = DecoderDeep(latent_dims, distance)
        elif structure == 'ising':
            self.encoder = VariationalEncoderIsing(latent_dims, distance, channels)
            self.decoder = DecoderIsing(latent_dims, distance, channels)
        elif structure == 'transformer':
            self.encoder = TransformerEncoder(latent_dims, distance, channels)
            self.decoder = TransformerDecoder(latent_dims, distance, channels)
        elif structure == 'pretrained':
            pass
        else:
            raise Exception('Invalid structure specified.')

    def forward(self, x):
        if self.structure == 'transformer' or self.structure == 'ising':
            # print(get_gpu_memory())
            # torch.cuda.synchronize()
            # start_time = time.time()
            z_mean, z_log_var, z = self.encoder(x)
            # torch.cuda.synchronize()
            # print("--- %s seconds encoding tot---" % (time.time() - start_time))
            # print(get_gpu_memory())
            # torch.cuda.synchronize()
            # start_time = time.time()
            x = self.decoder(z)
            # torch.cuda.synchronize()
            # print("--- %s seconds decoding tot---" % (time.time() - start_time))
            # print(get_gpu_memory())
        else:
            # print(get_gpu_memory())
            # start_time = time.time()
            z_mean, z_log_var, z, indices1, indices2, input_size1, input_size2 = self.encoder(x)
            # print("--- %s seconds ---" % (time.time() - start_time))
            # print(get_gpu_memory())
            # start_time = time.time()
            x = self.decoder(z, indices1, indices2, input_size1, input_size2)
            # print("--- %s seconds ---" % (time.time() - start_time))
            # print(get_gpu_memory())
        # TODO write differently using *args
        # TODO add values that should be used for skip connections a return values of the encoder to give them to
        # the decoder as additional parameters
        return x, z_mean, z_log_var


class VariationalAutoencoderPheno(Net):
    def __init__(self, latent_dims: int, distance: int, name: str, structure: str):
        super().__init__(name)
        self.encoder = VariationalEncoderPheno(latent_dims, distance)
