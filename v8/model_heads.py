import tensorflow as tf
from tensorflow import keras

def softplus_head(n_out: int, eps: float = 1e-12) -> keras.Sequential:
    """
    Dense → Softplus (strictly positive) → row-normalise so Σ = 1.
    Softplus keeps gradients alive for small logits, unlike ReLU.
    """
    return keras.Sequential(
        [
            keras.layers.Dense(n_out),                    # raw logits
            keras.layers.Activation(tf.nn.softplus),      # > 0 everywhere
            keras.layers.Lambda(
                lambda x: x / (tf.reduce_sum(x, axis=-1, keepdims=True) + eps),
                name="renorm"
            ),
        ],
        name="softplus_head"
    )
