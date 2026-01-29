import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

class Encoder(keras.Model):
    """
    Encoder class for the Convolutional Autoencoder.
    
    This class implements the encoder part of a convolutional autoencoder that
    compresses input images into a lower-dimensional latent representation.
    
    Input shape: (batch_size, height, width, channels) -> (batch_size, 1080, 1920, 3)
    Output shape: (batch_size, embedding_dim)
    
    The encoder uses three convolutional layers with decreasing spatial dimensions
    and increasing feature channels to extract hierarchical features from the input.
    
    Args:
        embedding_dim (int): Dimension of the latent space representation
        
    Forward pass:
        Input -> Conv1 (12x12 kernel, 6 stride) -> Conv2 (10x10 kernel, 5 stride) 
        -> Conv3 (4x4 kernel, 4 stride) -> Flatten -> Dense (embedding_dim)
    """
    def __init__(self, embedding_dim):
        super(Encoder, self).__init__()
        self.conv1 = layers.Conv2D(filters=16, kernel_size=8, strides=1, padding='same', activation='gelu')
        self.pool1 = layers.MaxPool2D(pool_size=5,padding='same')
        self.conv2 = layers.Conv2D(filters=32, kernel_size=8, strides=1, padding='same', activation='gelu')
        self.pool2 = layers.MaxPool2D(pool_size=4,padding='same')
        self.conv3 = layers.Conv2D(filters=64, kernel_size=4, strides=1, padding='same', activation='gelu')
        self.pool3 = layers.MaxPool2D(pool_size=3,padding='same')
        self.conv4 = layers.Conv2D(filters=128, kernel_size=2, strides=1, padding='same', activation='gelu')
        self.pool4 = layers.MaxPool2D(pool_size=2,padding='same')
        self.conv5_1 = layers.Conv2D(filters=256, kernel_size=2, strides=1, padding='same', activation='gelu')
        self.conv5_2 = layers.Conv2D(filters=128, kernel_size=2, strides=1, padding='same', activation='gelu')
        self.conv5_3 = layers.Conv2D(filters=32, kernel_size=2, strides=1, padding='same', activation='gelu')

        self.flatten = layers.Flatten()
        self.fc = layers.Dense(embedding_dim)

    def call(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.pool4(x)
        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class Decoder(keras.Model):
    """
    Decoder class for the Convolutional Autoencoder.
    
    This class implements the decoder part of a convolutional autoencoder that
    reconstructs images from their latent representations.
    
    Input shape: (batch_size, embedding_dim)
    Output shape: (batch_size, 1080, 1920, 3)
    
    The decoder uses three transposed convolutional layers with increasing spatial
    dimensions and decreasing feature channels to reconstruct the original image.
    
    Args:
        embedding_dim (int): Dimension of the latent space representation
        height (int): Height of the reshaped image
        width (int): Width of the reshaped image
        channels (int): Number of channels in the reshaped image
        
    Forward pass:
        Input -> Dense (height * width * channels) -> Reshape 
        -> Deconv1 (4x4 kernel, 4 stride) -> Deconv2 (10x10 kernel, 5 stride)
        -> Deconv3 (12x12 kernel, 6 stride)
    """
    def __init__(self, embedding_dim, height, width, channels):
        super(Decoder, self).__init__()
        self.fc = layers.Dense(height * width * channels)
        self.target_shape = (height, width, channels)
        self.reshape = layers.Reshape(self.target_shape)
        self.conv1_1 = layers.Conv2D(filters=32, kernel_size=2, strides=1, padding='same', activation='gelu')
        self.conv1_2 = layers.Conv2D(filters=32, kernel_size=2, strides=1, padding='same', activation='gelu')
        self.conv1_3 = layers.Conv2D(filters=32, kernel_size=2, strides=1, padding='same', activation='gelu')
        self.upSample1 = layers.UpSampling2D(size=2,interpolation='bilinear')
        self.conv2_1 = layers.Conv2D(filters=32, kernel_size=2, strides=1, padding='same', activation='gelu')
        self.conv2_2 = layers.Conv2D(filters=64, kernel_size=2, strides=1, padding='same', activation='gelu')
        self.upSample2 = layers.UpSampling2D(size=3,interpolation='bilinear')
        self.conv3_1 = layers.Conv2D(filters=64, kernel_size=4, strides=1, padding='same', activation='gelu')
        self.conv3_2 = layers.Conv2D(filters=32, kernel_size=4, strides=1, padding='same', activation='gelu')
        self.upSample3 = layers.UpSampling2D(size=4,interpolation='bilinear')
        self.conv4_1 = layers.Conv2D(filters=32, kernel_size=8, strides=1, padding='same', activation='gelu')
        self.conv4_2 = layers.Conv2D(filters=16, kernel_size=8, strides=1, padding='same', activation='gelu')
        self.upSample4 = layers.UpSampling2D(size=5,interpolation='bilinear')
        self.conv5_1 = layers.Conv2D(filters=16, kernel_size=8, strides=1, padding='same', activation='gelu')
        self.conv5_2 = layers.Conv2D(filters=8, kernel_size=4, strides=1, padding='same', activation='gelu')
        self.conv5_3 = layers.Conv2D(filters=3, kernel_size=4, strides=1, padding='same', activation='sigmoid')

    def call(self, x):
        x = self.fc(x)
        x = self.reshape(x)
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.conv1_3(x)
        x = self.upSample1(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.upSample2(x)
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.upSample3(x)
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.upSample4(x)
        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        return x


class ConvAutoencoder(keras.Model):
    """
    Convolutional Autoencoder model for image reconstruction.
    
    This class implements a complete convolutional autoencoder with an encoder
    that compresses images into a latent space representation, and a decoder
    that reconstructs images from this representation.
    
    The model architecture:
    - Encoder: 3 Conv2D layers with decreasing spatial dimensions
    - Latent space: Dense layer with specified embedding dimension
    - Decoder: 3 Conv2DTranspose layers with increasing spatial dimensions
    
    Input shape: (batch_size, 1080, 1920, 3)
    Embedding shape: (batch_size, embedding_dim)
    Output shape: (batch_size, 1080, 1920, 3)
    
    Args:
        embed_dim (int): Dimension of the latent space representation (default: 64)
        
    Forward pass:
        Input -> Encoder -> Latent vector -> Decoder -> Reconstructed image
    """
    def __init__(self, embed_dim):
        super(ConvAutoencoder, self).__init__()
        self.encoder = Encoder(embed_dim)
        self.decoder = Decoder(embed_dim, height=9, width=16, channels=32)

    def call(self, x):
        latent_vector = self.encoder(x)
        reconstructed_image = self.decoder(latent_vector)
        return reconstructed_image
