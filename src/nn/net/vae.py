from .net import *
from .encoder import *
from .decoder import *


class VariationalAutoencoder(Net):
    def __init__(self, latent_dims: int, distance: int, name: str, structure: str, noise: str):
        super(VariationalAutoencoder, self).__init__(name)
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
        else:
            raise Exception('Invalid structure specified.')

    def forward(self, x):
        z_mean, z_log_var, z, indices1, indices2, input_size1, input_size2 = self.encoder(x)
        # TODO add values that should be used for skip connections a return values of the encoder to give them to
        # the decoder as additional parameters
        x = self.decoder(z, indices1, indices2, input_size1, input_size2)  # TODO write differently using *args
        return x, z_mean, z_log_var


class VariationalAutoencoderPheno(Net):
    def __init__(self, latent_dims: int, distance: int, name: str, structure: str):
        super().__init__(name)
        self.encoder = VariationalEncoderPheno(latent_dims, distance)
