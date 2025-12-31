import numpy as np
from model import ConvAutoencoder

x_train = np.load("../preprocess/dataset.npy")

model = ConvAutoencoder(embed_dim=64, batch_size=4)
pred = model(x_train)

print("Encoder summary:")
model.encoder.summary()
print("\nDecoder summary:")
model.decoder.summary()

model.compile(optimizer='adam', loss='mse')
model.fit(x_train, x_train, epochs=10, batch_size=4)

