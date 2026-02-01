import numpy as np
import glob
import tensorflow as tf

def do_training(model, train_dataset, val_dataset, epochs, batch_size, out_path, train_size, val_size):
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
    model.fit(train_dataset, epochs=epochs, validation_data=val_dataset, steps_per_epoch=train_size // batch_size, validation_steps=val_size // batch_size)

    # Save the weights
    model.save_weights(out_path+"model.weights.h5")

if __name__=="__main__":
    from model import ConvAutoencoder
    from data_loader import create_dataset

    # Parameters
    model = ConvAutoencoder(embed_dim=64, channels=16)
    path="../output_debug/"
    epochs=2
    batch_size=32

    # Load the dataset
    train_file_paths = sorted(glob.glob(path+'Train/preprocessed*.npy'))
    val_file_paths = sorted(glob.glob(path+'Val/preprocessed*.npy'))
    train_dataset = create_dataset(train_file_paths, batch_size=batch_size, shuffle=True)
    val_dataset = create_dataset(val_file_paths, batch_size=batch_size, shuffle=False)

    do_training(model, train_dataset, val_dataset, epochs, batch_size, path, len(train_file_paths), len(val_file_paths))
