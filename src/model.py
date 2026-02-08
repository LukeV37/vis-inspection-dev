import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

class Encoder(keras.Model):
    """
    Encoder class for the Convolutional Autoencoder.

    This class implements the encoder part of a convolutional autoencoder that
    compresses input images into a lower-dimensional latent representation.

    Input shape: (batch_size, 384, 384, 3)
    Output shape: (batch_size, embedding_dim)

    The encoder uses 8 convolutional layers with progressive downsampling:
    - First layer: downsample by factor of 3 (384 -> 128)
    - Next 7 layers: downsample by factor of 2 each (128 -> 64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1)

    Args:
        embedding_dim (int): Dimension of the latent space representation
    """
    def __init__(self, embedding_dim, channels):
        super(Encoder, self).__init__()

        # Single activation layer to reuse
        self.leaky_relu = layers.LeakyReLU(alpha=0.2)

        # Conv1: (batch, 384, 384, 3) -> (batch, 128, 128, 32)
        self.conv1 = layers.Conv2D(filters=32, kernel_size=3, strides=3, padding='same')
        self.bn1 = layers.BatchNormalization()

        # Conv2: (batch, 128, 128, 32) -> (batch, 64, 64, 64)
        self.conv2 = layers.Conv2D(filters=64, kernel_size=3, strides=2, padding='same')
        self.bn2 = layers.BatchNormalization()

        # Conv3: (batch, 64, 64, 64) -> (batch, 32, 32, 128)
        self.conv3 = layers.Conv2D(filters=128, kernel_size=3, strides=2, padding='same')
        self.bn3 = layers.BatchNormalization()

        # Conv4: (batch, 32, 32, 128) -> (batch, 16, 16, 256)
        self.conv4 = layers.Conv2D(filters=256, kernel_size=3, strides=2, padding='same')
        self.bn4 = layers.BatchNormalization()

        # Conv5: (batch, 16, 16, 256) -> (batch, 8, 8, 512)
        self.conv5 = layers.Conv2D(filters=512, kernel_size=3, strides=2, padding='same')
        self.bn5 = layers.BatchNormalization()

        # Conv6: (batch, 8, 8, 512) -> (batch, 4, 4, 512)
        self.conv6 = layers.Conv2D(filters=512, kernel_size=3, strides=2, padding='same')
        self.bn6 = layers.BatchNormalization()

        # Conv7: (batch, 4, 4, 512) -> (batch, 2, 2, 512)
        self.conv7 = layers.Conv2D(filters=1024, kernel_size=3, strides=2, padding='same')
        self.bn7 = layers.BatchNormalization()

        # Conv8: (batch, 2, 2, 512) -> (batch, 1, 1, channels)
        self.conv8 = layers.Conv2D(filters=channels, kernel_size=3, strides=2, padding='same')
        self.bn8 = layers.BatchNormalization()

        # Flatten: (batch, 1, 1, channels) -> (batch, channels)
        self.flatten = layers.Flatten()

        # Fully Connected: (batch, channels) -> (batch, embedding_dim)
        self.fc = layers.Dense(embedding_dim)

    def call(self, x, training=False):
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.leaky_relu(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.leaky_relu(x)

        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = self.leaky_relu(x)

        x = self.conv4(x)
        x = self.bn4(x, training=training)
        x = self.leaky_relu(x)

        x = self.conv5(x)
        x = self.bn5(x, training=training)
        x = self.leaky_relu(x)

        x = self.conv6(x)
        x = self.bn6(x, training=training)
        x = self.leaky_relu(x)

        x = self.conv7(x)
        x = self.bn7(x, training=training)
        x = self.leaky_relu(x)

        x = self.conv8(x)
        x = self.bn8(x, training=training)
        x = self.leaky_relu(x)

        x = self.flatten(x)
        x = self.fc(x)
        return x


class Decoder(keras.Model):
    """
    Decoder class for the Convolutional Autoencoder.

    This class implements the decoder part of a convolutional autoencoder that
    reconstructs images from their latent representations.

    Input shape: (batch_size, embedding_dim)
    Output shape: (batch_size, 384, 384, 3)

    The decoder uses 8 transposed convolutional layers with progressive upsampling:
    - First 7 layers: upsample by factor of 2 each (1 -> 2 -> 4 -> 8 -> 16 -> 32 -> 64 -> 128)
    - Last layer: upsample by factor of 3 (128 -> 384)

    Args:
        embedding_dim (int): Dimension of the latent space representation
    """
    def __init__(self, embedding_dim, channels):
        super(Decoder, self).__init__()

        # Single activation layer to reuse
        self.leaky_relu = layers.LeakyReLU(alpha=0.2)

        # Dense layer: (batch, embedding_dim) -> (batch, 512)
        self.fc = layers.Dense(channels)

        # Reshape layer: (batch, 512) -> (batch, 1, 1, 512)
        self.reshape = layers.Reshape((1, 1, channels))

        # Deconv1: (batch, 1, 1, 512) -> (batch, 2, 2, 512)
        self.deconv1 = layers.Conv2DTranspose(filters=512, kernel_size=3, strides=2, padding='same')
        self.bn1 = layers.BatchNormalization()

        # Deconv2: (batch, 2, 2, 512) -> (batch, 4, 4, 512)
        self.deconv2 = layers.Conv2DTranspose(filters=512, kernel_size=3, strides=2, padding='same')
        self.bn2 = layers.BatchNormalization()

        # Deconv3: (batch, 4, 4, 512) -> (batch, 8, 8, 512)
        self.deconv3 = layers.Conv2DTranspose(filters=512, kernel_size=3, strides=2, padding='same')
        self.bn3 = layers.BatchNormalization()

        # Deconv4: (batch, 8, 8, 512) -> (batch, 16, 16, 256)
        self.deconv4 = layers.Conv2DTranspose(filters=256, kernel_size=3, strides=2, padding='same')
        self.bn4 = layers.BatchNormalization()

        # Deconv5: (batch, 16, 16, 256) -> (batch, 32, 32, 128)
        self.deconv5 = layers.Conv2DTranspose(filters=128, kernel_size=3, strides=2, padding='same')
        self.bn5 = layers.BatchNormalization()

        # Deconv6: (batch, 32, 32, 128) -> (batch, 64, 64, 64)
        self.deconv6 = layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same')
        self.bn6 = layers.BatchNormalization()

        # Deconv7: (batch, 64, 64, 64) -> (batch, 128, 128, 32)
        self.deconv7 = layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same')
        self.bn7 = layers.BatchNormalization()

        # Deconv8: (batch, 128, 128, 32) -> (batch, 384, 384, 3)
        self.deconv8 = layers.Conv2DTranspose(filters=16, kernel_size=3, strides=3, padding='same')

        # Refinement layer 4 (no upsampling):
        self.deconv9 = layers.Conv2DTranspose(filters=8, kernel_size=3, strides=1, padding='same')

        # Output reconstruction layer:
        self.deconv10 = layers.Conv2DTranspose(filters=3, kernel_size=3, strides=1, padding='same', activation='sigmoid')

    def call(self, x, training=False):
        x = self.fc(x)
        x = self.reshape(x)

        x = self.deconv1(x)
        x = self.bn1(x, training=training)
        x = self.leaky_relu(x)

        x = self.deconv2(x)
        x = self.bn2(x, training=training)
        x = self.leaky_relu(x)

        x = self.deconv3(x)
        x = self.bn3(x, training=training)
        x = self.leaky_relu(x)

        x = self.deconv4(x)
        x = self.bn4(x, training=training)
        x = self.leaky_relu(x)

        x = self.deconv5(x)
        x = self.bn5(x, training=training)
        x = self.leaky_relu(x)

        x = self.deconv6(x)
        x = self.bn6(x, training=training)
        x = self.leaky_relu(x)

        x = self.deconv7(x)
        x = self.bn7(x, training=training)
        x = self.leaky_relu(x)

        x = self.deconv8(x)
        x = self.leaky_relu(x)
        x = self.deconv9(x)
        x = self.leaky_relu(x)
        x = self.deconv10(x)
        return x


class ConvAutoencoder(keras.Model):
    """
    Convolutional Autoencoder model for image reconstruction.

    This class implements a complete convolutional autoencoder with an encoder
    that compresses images into a latent space representation, and a decoder
    that reconstructs images from this representation.

    The model architecture follows a symmetric design:
    - Encoder: 1 conv layer (3x downsample) + 7 conv layers (2x downsample each)
    - Latent space: Dense layer with specified embedding dimension
    - Decoder: 7 deconv layers (2x upsample each) + 1 deconv layer (3x upsample)

    Input shape: (batch_size, 384, 384, 3)
    Embedding shape: (batch_size, embedding_dim)
    Output shape: (batch_size, 384, 384, 3)

    Features for stable training:
    - Batch Normalization after each conv/deconv layer
    - LeakyReLU activations to prevent dying gradients
    - Progressive channel scaling (32 -> 64 -> 128 -> 256 -> 512)

    Args:
        embed_dim (int): Dimension of the latent space representation (default: 64)
    """
    def __init__(self, embed_dim=64, channels=512):
        super(ConvAutoencoder, self).__init__()
        self.encoder = Encoder(embed_dim, channels)
        self.decoder = Decoder(embed_dim, channels)

    def call(self, x, training=False):
        latent_vector = self.encoder(x, training=training)
        reconstructed_image = self.decoder(latent_vector, training=training)
        return reconstructed_image
