import numpy as np
import glob
from data_loader import create_dataset
import tensorflow as tf

def do_training(in_path, model_file, model, epochs, batch_size):
    # Load the dataset
    train_file_paths = sorted(glob.glob(in_path+'Train/preprocessed*.npy'))
    val_file_paths = sorted(glob.glob(in_path+'Val/preprocessed*.npy'))

    train_dataset = create_dataset(train_file_paths, batch_size=batch_size, shuffle=True)
    val_dataset = create_dataset(val_file_paths, batch_size=batch_size, shuffle=False)

    # Pass dummy data to initialize the model
    dummy_input = tf.zeros((1, 1080, 1920, 3))
    _ = model(dummy_input)

    # Print Summary of the model
    print("Encoder summary:")
    model.encoder.summary()
    print("\nDecoder summary:")
    model.decoder.summary()

    # Compile the model
    model.compile(optimizer='adam', loss='mse')

    # Train the model
    model.fit(train_dataset, epochs=epochs, validation_data=val_dataset, steps_per_epoch=len(train_file_paths) // batch_size, validation_steps=len(val_file_paths) // batch_size)

    # Save the weights
    model.save_weights(model_file)

if __name__=="__main__":
    from model import ConvAutoencoder

    model = ConvAutoencoder(embed_dim=64, channels=16)
    model_file="../output/my_model.weights.h5"
    in_path="../output/"
    epochs=2
    batch_size=32
    do_training(in_path, model_file, model, epochs, batch_size)
