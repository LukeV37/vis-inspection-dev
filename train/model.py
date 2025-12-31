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
        # Conv1: Input (batch, 1080, 1920, 3) -> Output (batch, 180, 320, 4)
        self.conv1 = layers.Conv2D(4, kernel_size=12, strides=6, padding='same', activation='relu')
        # Conv2: Input (batch, 180, 320, 4) -> Output (batch, 36, 64, 8)
        self.conv2 = layers.Conv2D(8, kernel_size=10, strides=5, padding='same', activation='relu')
        # Conv3: Input (batch, 36, 64, 8) -> Output (batch, 9, 16, 16)
        self.conv3 = layers.Conv2D(16, kernel_size=4, strides=4, padding='same', activation='relu')
        # Flatten: Input (batch, 9, 16, 16) -> Output (batch, 2304)
        self.flatten = layers.Flatten()
        # Fully Connected: Input (batch, 9*16*16) -> Output (batch, embedding_dim)
        self.fc = layers.Dense(embedding_dim)

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
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
        # Dense layer: Input (batch, embedding_dim) -> Output (batch, 9*16*16)
        self.fc = layers.Dense(height * width * channels)
        # Reshape layer: Input (batch, 9*16*16) -> Output (batch, 9, 16, 16)
        self.target_shape = (height, width, channels)
        self.reshape = layers.Reshape(self.target_shape)
        # Deconv1: Input (batch, 9, 16, 16) -> Output (batch, 36, 64, 8)
        self.deconv1 = layers.Conv2DTranspose(8, kernel_size=4, strides=4, padding='same', activation='relu')
        # Deconv2: Input (batch, 36, 64, 8) -> Output (batch, 180, 320, 4)
        self.deconv2 = layers.Conv2DTranspose(4, kernel_size=10, strides=5, padding='same', activation='relu')
        # Deconv3: Input (batch, 180, 320, 4) -> Output (batch, 1080, 1920, 3)
        self.deconv3 = layers.Conv2DTranspose(3, kernel_size=12, strides=6, padding='same', activation='sigmoid')

    def call(self, x):
        x = self.fc(x)
        x = self.reshape(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
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
    def __init__(self, embed_dim=64):
        super(ConvAutoencoder, self).__init__()
        self.encoder = Encoder(embed_dim)
        self.decoder = Decoder(embed_dim, height=9, width=16, channels=16)

    def call(self, x):
        latent_vector = self.encoder(x)
        reconstructed_image = self.decoder(latent_vector)
        return reconstructed_image
