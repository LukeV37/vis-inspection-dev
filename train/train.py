#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


# In[2]:


class Encoder(nn.Module):
    def __init__(self, image_height, image_width, embedding_dim):
        super(Encoder, self).__init__()
        # define convolutional layers
        self.conv1 = nn.Conv2d(3, 4, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(4, 6, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(6, 8, kernel_size=3, stride=2, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # variable to store the shape of the output tensor before flattening
        # the features, it will be used in decoders input while reconstructing
        self.shape_before_flattening = None
        # compute the flattened size after convolutions
        flattened_size = (image_height // 8) * (image_width // 8) * 8
        # define fully connected layer to create embeddings
        self.fc = nn.Linear(flattened_size, embedding_dim)
    def forward(self, x):
        #print("Input Shape:\t", x.shape)
        # apply ReLU activations after each convolutional layer
        x = F.relu(self.conv1(x))
        #print("First Conv:\t", x.shape)
        #x = self.pool(x)
        #print("First Pool:\t", x.shape)
        x = F.relu(self.conv2(x))
        #print("Second Conv:\t", x.shape)
        #x = self.pool(x)
        #print("Second Pool:\t", x.shape)
        x = F.relu(self.conv3(x))
        #print("Third Conv:\t", x.shape)
        #x = self.pool(x)
        #print("Third Pool:\t", x.shape)
        # store the shape before flattening
        self.shape_before_flattening = x.shape
        # flatten the tensor
        x = torch.flatten(x, start_dim=1)
        #print("Flatten Shape:\t", x.shape)
        # apply fully connected layer to generate embeddings
        x = self.fc(x)
        return x


# In[3]:


class Decoder(nn.Module):
    def __init__(self, embedding_dim, shape_before_flattening):
        super(Decoder, self).__init__()
        # define fully connected layer to unflatten the embeddings
        self.fc = nn.Linear(embedding_dim, np.prod(shape_before_flattening[1:]))
        # store the shape before flattening
        self.reshape_dim = shape_before_flattening
        # define transpose convolutional layers
        self.deconv1 = nn.ConvTranspose2d(8, 6, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(6, 4, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(4, 3, kernel_size=3, stride=2, padding=1, output_padding=1)
    def forward(self, x):
        #print("Input Shape:\t", x.shape)
        # apply fully connected layer to unflatten the embeddings
        x = self.fc(x)
        #print("First FC:\t", x.shape)
        # reshape the tensor to match shape before flattening
        x = x.view(*self.reshape_dim)
        #x = x.view(x.size(0), *self.reshape_dim)
        #print("Reshape:\t", x.shape)
        # apply ReLU activations after each transpose convolutional layer
        x = F.relu(self.deconv1(x))
        #print("First DeConv:\t", x.shape)
        x = F.relu(self.deconv2(x))
        #print("Second DeConv:\t", x.shape)
        x = F.relu(self.deconv3(x))
        #print("Third DeConv:\t", x.shape)
        # apply sigmoid activation to the final convolutional layer to generate output image
        x = torch.sigmoid(x)
        return x


# In[4]:


class ConvAutoencoder(nn.Module):
    def __init__(self, in_channels=3, embed_dim=64, batchsize=4):
        super(ConvAutoencoder, self).__init__()
        self.batchsize=batchsize
        self.encoder = Encoder(1080, 1920, embed_dim)
        self.decoder = Decoder(embed_dim, [batchsize, 8, 135, 240])

    def forward(self, x):
        latent_vector = self.encoder(x)
        reconstructed_image = self.decoder(latent_vector)
        return reconstructed_image


# In[5]:


dataset = np.load("out.npy")


# In[6]:


dataset_reshaped = np.transpose(dataset[:,:,:,0:3], (0, 3, 1, 2))
input_data = torch.Tensor(dataset_reshaped)


# In[7]:


class ImageDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        return image


# In[8]:


training_data=ImageDataset(input_data[0:400])
test_data=ImageDataset(input_data[400:])


# In[9]:


def train(num_epochs=10):
    for epoch in range(num_epochs):
        for batch_images in train_loader:
            batch_images = batch_images.to(device)

            optimizer.zero_grad()

            outputs = model(batch_images)
            loss = criterion(outputs, batch_images)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 1 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


# In[10]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
model = ConvAutoencoder(batchsize=4).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Placeholder for actual data loading
train_loader = torch.utils.data.DataLoader(
    training_data,
    batch_size=model.batchsize,
    shuffle=True
)


# In[11]:


print("Trainable Params: ", sum(p.numel() for p in model.parameters() if p.requires_grad))


# In[12]:


train()


# In[20]:


# Placeholder for actual data loading
test_loader = torch.utils.data.DataLoader(
    test_data,
    batch_size=model.batchsize,
    shuffle=True
)

for batch_images in test_loader:
    pred = model(batch_images.to(device))
    print(pred.shape)
print()


# In[ ]:




