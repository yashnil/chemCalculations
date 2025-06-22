
# model_heads.py ---------------------------------------------------------
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Lambda, ReLU, Activation

def softmax_T(n_out: int, T: float = 0.5):
    """Soft-max with temperature scaling (<1) – stripe-free but still normalises."""
    return keras.Sequential([
        keras.layers.Dense(n_out, name="raw_logits"),
        Lambda(lambda x: x / T, name=f"temp_scale_T{T:g}"),
        Activation("softmax", name="softmax_T")
    ], name="head_Tsoftmax")

def relu_renorm(n_out: int):
    """ReLU → row-sum renormalisation – no coupling at all."""
    return keras.Sequential([
        keras.layers.Dense(n_out, name="raw_logits"),
        ReLU(name="relu_nonneg"),
        Lambda(lambda x: x / (tf.reduce_sum(x, 1, keepdims=True) + 1e-12),
               name="renorm")
    ], name="head_relu_norm")
# -----------------------------------------------------------------------
