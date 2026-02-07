import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


class Encoder(keras.Model):
    def __init__(self, embedding_dim, channels):
        super(Encoder, self).__init__()

        self.conv1 = layers.Conv2D(4, 3, strides=1, padding='same', activation='relu')
        self.pool1 = layers.MaxPool2D(pool_size=2, padding='same')

        self.conv2 = layers.Conv2D(8, 3, strides=1, padding='same', activation='relu')
        self.pool2 = layers.MaxPool2D(pool_size=2, padding='same')

        self.conv3 = layers.Conv2D(channels, 3, strides=1, padding='same', activation='relu')
        self.pool3 = layers.MaxPool2D(pool_size=2, padding='same')

        self.flatten = layers.Flatten()
        self.fc = layers.Dense(embedding_dim)

    def call(self, x):
        x = self.conv1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.pool3(x)

        x = self.flatten(x)
        x = self.fc(x)
        return x


class Decoder(keras.Model):
    def __init__(self, embedding_dim, height, width, channels):
        super(Decoder, self).__init__()

        self.fc = layers.Dense(height * width * channels)
        self.reshape = layers.Reshape((height, width, channels))

        self.deconv1 = layers.Conv2DTranspose(16, 3, strides=2, padding='same', activation='relu')
        self.deconv2 = layers.Conv2DTranspose(16, 3, strides=2, padding='same', activation='relu')
        self.deconv3 = layers.Conv2DTranspose(16, 3, strides=2, padding='same', activation='relu')

        self.deconv4 = layers.Conv2DTranspose(8, 3, strides=1, padding='same', activation='relu')
        self.deconv5 = layers.Conv2DTranspose(3, 3, strides=1, padding='same', activation='sigmoid')

    def call(self, x):
        x = self.fc(x)
        x = self.reshape(x)

        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)

        x = self.deconv4(x)
        x = self.deconv5(x)
        return x


class ConvAutoencoder(keras.Model):
    def __init__(self, embed_dim=64, channels=8):
        super(ConvAutoencoder, self).__init__()

        self.encoder = Encoder(embed_dim, channels=channels)

        # 1080 / 2 / 2 / 2 = 135
        # 1920 / 2 / 2 / 2 = 240
        self.decoder = Decoder(embed_dim, height=135, width=240, channels=channels)

    def call(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed
