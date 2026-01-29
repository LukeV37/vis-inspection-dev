import numpy as np

def do_training(in_path, model_file, model, epochs):
    # Load the dataset
    dataset = np.load(in_path)

    # Determine train/test split base on number of samples
    num_samples=len(dataset)
    train_split=int(0.75*num_samples)
    test_split=int(0.8*num_samples)

    # Split the dataset into train/val/test
    x_train = dataset[:train_split]
    x_val   = dataset[train_split:test_split]
    x_test  = dataset[test_split:]

    # Pass sample data to initialize the model
    pred_image = model(x_train[0:1])

    # Print Summary of the model
    print("Encoder summary:")
    model.encoder.summary()
    print("\nDecoder summary:")
    model.decoder.summary()

    # Compile the model
    model.compile(optimizer='adam', loss='mse')

    # Train the model
    model.fit(x_train, x_train, epochs=epochs, batch_size=4, validation_data=(x_val, x_val))

    # Save the weights
    model.save_weights(model_file)


if __name__=="__main__":
    from model import ConvAutoencoder
    model_file="../output/my_model.weights.h5"
    model = ConvAutoencoder(embed_dim=64)
    in_path = "../output/dataset.npy"
    epochs=2
    do_training(in_path, model_file, model, epochs)
