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
    Output shape: (batch_size, embedding_dim) after flattening and dense projection

   
   three convolutional blocks with max pooling that progressively reduce spatial dimensions
   and increasing feature channels to extract hierarchical features from the input.
   
    Args:
        embedding_dim (int): Dimension of the latent space representation

    Forward pass:
      Input -> Conv1 (3x3) -> MaxPool (2x2)
      -> Conv2 (3x3) -> MaxPool (2x2)
      -> Conv3 (3x3) -> MaxPool (2x2)
      -> Flatten -> Dense (embedding_dim)

    """
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
    """
    Decoder class for the Convolutional Autoencoder.
   
    This class implements the decoder part of a convolutional autoencoder that
    reconstructs images from their latent representations.
   
    Input shape: (batch_size, embedding_dim)
    Output shape: (batch_size, 1080, 1920, 3)
   
    The decoder uses five transposed convolutional layers: three for upsampling
    and two for feature refinement and output reconstruction.

   
    Args:
        embedding_dim (int): Dimension of the latent space representation
        height (int): Height of the reshaped image
        width (int): Width of the reshaped image
        channels (int): Number of channels in the reshaped image
       
 
    Forward pass:
    -> Deconv (3x3, stride 2) x3  # spatial upsampling
    -> Deconv (3x3, stride 1)    # feature refinement
    -> Deconv (3x3, stride 1)    # RGB reconstruction
    """
   
    def __init__(self, embedding_dim, height, width, channels):
        super(Decoder, self).__init__()

        self.fc = layers.Dense(height * width * channels)
        self.reshape = layers.Reshape((height, width, channels))

        
        # Upsampling layer 1:
        # Doubles the spatial resolution from 135x240 → 270x480.
        # Learns coarse spatial structure while expanding feature maps.
        self.deconv1 = layers.Conv2DTranspose(16, 3, strides=2, padding='same', activation='relu')
        
        
        
        # Upsampling layer 2:
        # Doubles the spatial resolution from 270x480 → 540x960.
        # Refines larger shapes and object layout at mid-level resolution.
        self.deconv2 = layers.Conv2DTranspose(16, 3, strides=2, padding='same', activation='relu')
        
        
        
        # Upsampling layer 3:
        # Doubles the spatial resolution from 540x960 → 1080x1920 (original image size).
        # Restores full image geometry and global spatial alignment.
        self.deconv3 = layers.Conv2DTranspose(16, 3, strides=2, padding='same', activation='relu')

        
        # Refinement layer 4 (no upsampling):
        # Keeps spatial resolution fixed at 1080x1920.
        # Learns local texture, edge continuity, and reduces upsampling artifacts.
        self.deconv4 = layers.Conv2DTranspose(8, 3, strides=1, padding='same', activation='relu')
        
        
        
        # Output reconstruction layer:
        # Keeps spatial resolution fixed and maps feature channels to RGB.
        # Produces the final reconstructed image with pixel values in [0, 1].
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
    """
    Convolutional Autoencoder model for image reconstruction.
   
    This class implements a complete convolutional autoencoder with an encoder
    that compresses images into a latent space representation, and a decoder
    that reconstructs images from this representation.
   
    The model architecture:
    - Encoder: 3 Conv2D layers with decreasing spatial dimensions
    - Latent space: Dense layer with specified embedding dimension
    - Decoder: 5 Conv2DTranspose layers with increasing spatial dimensions
   
    Input shape: (batch_size, 1080, 1920, 3)
    Embedding shape: (batch_size, embedding_dim)
    Output shape: (batch_size, 1080, 1920, 3)
   
    Args:
        embed_dim (int): Dimension of the latent space representation (default: 64)
       
    Forward pass:
        Input -> Encoder -> Latent vector -> Decoder -> Reconstructed image
    """
    
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
