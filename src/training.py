import numpy as np
import glob
import tensorflow as tf
import matplotlib.pyplot as plt

def do_training(model, train_dataset, val_dataset, epochs, batch_size, out_path, train_size, val_size):
    # Pass dummy data to initialize the model
    dummy_input = tf.zeros((1, 384, 384, 3))
    _ = model(dummy_input)

    # Print Summary of the model
    print("Encoder summary:")
    model.encoder.summary()
    print("\nDecoder summary:")
    model.decoder.summary()

    # Define combined loss for better reconstruction
    def combined_loss(y_true, y_pred):
        # MAE for sharper reconstruction
        mae = tf.reduce_mean(tf.abs(y_true - y_pred))
        # SSIM for structural similarity
        ssim_loss = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
        # Weighted: prioritize MAE slightly
        return 0.6 * mae + 0.4 * ssim_loss

    # Use Adam with learning rate schedule
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-4,
        decay_steps=train_size // batch_size * 5,  # Decay every 5 epochs
        decay_rate=0.9
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    # Compile the model
    model.compile(optimizer=optimizer, loss=combined_loss, metrics=['mae'])

    # Train the model
    history = model.fit(train_dataset, epochs=epochs, validation_data=val_dataset, steps_per_epoch=train_size // batch_size, validation_steps=val_size // batch_size)

    # Save the weights
    model.save_weights(out_path+"model.weights.h5")

    # Plot training history
    plt.figure(figsize=(12, 4))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot MAE
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(out_path + 'training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\nTraining complete!")
    print(f"Final training loss: {history.history['loss'][-1]:.4f}")
    print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")
    print(f"Final training MAE: {history.history['mae'][-1]:.4f}")
    print(f"Final validation MAE: {history.history['val_mae'][-1]:.4f}")

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
