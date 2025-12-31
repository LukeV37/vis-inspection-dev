import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

class Encoder(keras.Model):
    def __init__(self, embedding_dim):
        super(Encoder, self).__init__()
        self.conv1 = layers.Conv2D(4, kernel_size=12, strides=6, padding='same', activation='relu')
        self.conv2 = layers.Conv2D(8, kernel_size=10, strides=5, padding='same', activation='relu')
        self.conv3 = layers.Conv2D(16, kernel_size=4, strides=4, padding='same', activation='relu')
        self.flatten = layers.Flatten()
        self.fc = layers.Dense(embedding_dim)

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class Decoder(keras.Model):
    def __init__(self, embedding_dim, height, width, channels):
        super(Decoder, self).__init__()
        self.fc = layers.Dense(height * width * channels)
        self.target_shape = (height, width, channels)
        self.reshape = layers.Reshape(self.target_shape)
        self.deconv1 = layers.Conv2DTranspose(8, kernel_size=4, strides=4, padding='same', activation='relu')
        self.deconv2 = layers.Conv2DTranspose(4, kernel_size=10, strides=5, padding='same', activation='relu')
        self.deconv3 = layers.Conv2DTranspose(3, kernel_size=12, strides=6, padding='same', activation='sigmoid')

    def call(self, x):
        x = self.fc(x)
        x = self.reshape(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        return x


class ConvAutoencoder(keras.Model):
    def __init__(self, in_channels=3, embed_dim=64, batch_size=4):
        super(ConvAutoencoder, self).__init__()
        self.batch_size = batch_size
        self.encoder = Encoder(embed_dim)
        self.decoder = Decoder(embed_dim, height=9, width=16, channels=16)

    def call(self, x):
        latent_vector = self.encoder(x)
        reconstructed_image = self.decoder(latent_vector)
        return reconstructed_image
