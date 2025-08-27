import tensorflow as tf
from tensorflow import keras

def softplus_head(n_out: int, gamma: float = 1.5, eps: float = 1e-12):
    """
    Softplus → power(γ) → renormalize.  γ≥1 sharpens the distribution and
    suppresses the ~1/N 'equipartition' shelf around ~1e-2.
    """
    return keras.Sequential(
        [
            keras.layers.Dense(n_out, activation="softplus", name="head_dense"),
            keras.layers.Lambda(lambda t, g=gamma, e=eps: tf.pow(t + e, g),
                                name="sharpen_pow"),
            keras.layers.Lambda(lambda t, e=eps: t / (tf.reduce_sum(t, axis=1, keepdims=True) + e),
                                name="norm_sum1"),
        ],
        name=f"softplus_head_gamma_{gamma}"
    )