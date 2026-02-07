import tensorflow as tf
import numpy as np

def create_dataset(file_paths, batch_size=32, shuffle=True, shuffle_buffer_size=1000):
    """
    Create a TensorFlow dataset from numpy file paths.

    Args:
        file_paths: List of paths to .npy files
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle the dataset
        shuffle_buffer_size: Size of shuffle buffer (larger = more random but more memory)

    Returns:
        tf.data.Dataset ready for training
    """

    def load_npy_file(filepath):
        """Load a single .npy file"""
        filepath = filepath.numpy().decode('utf-8')
        image = np.load(filepath)/255 # Normalize inputs
        return image.astype(np.float32) # Convert to float

    def tf_load_npy(filepath):
        """Wrapper to use numpy loading in tf.data pipeline"""
        image = tf.py_function(
            func=load_npy_file,
            inp=[filepath],
            Tout=tf.float32
        )
        image.set_shape([384, 384, 3])
        return image, image

    # Create dataset from file paths
    dataset = tf.data.Dataset.from_tensor_slices(file_paths)

    # Shuffle file paths if requested
    if shuffle:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)

    # Load images on-the-fly
    dataset = dataset.map(tf_load_npy, num_parallel_calls=tf.data.AUTOTUNE)

    # Batch the dataset
    dataset = dataset.batch(batch_size)

    # Prefetch for performance
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset
