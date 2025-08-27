
# losses.py  – custom loss functions for the FastChem surrogate
# -------------------------------------------------------------
#  * balanced_KL : emphasises rare species by weighting each KL
#                  term with 1 / ⟨abundance⟩
#  * mae_log     : mean-absolute-error in log₁₀ space
#  * composite_loss(lam) = lam·balanced_KL  +  (1-lam)·mae_log
#
# Recommended λ  : 0.6  (good compromise between linear MAE and
#                               log-space parity)
# -------------------------------------------------------------

import tensorflow as tf

EPS   = 1e-12
LOG10 = tf.math.log(10.0)

def _balanced_kl(y_true, y_pred):
    w = 1.0 / (tf.reduce_mean(y_true, axis=0) + EPS)
    w = w / tf.reduce_sum(w)
    kl_elem = y_true * (tf.math.log(y_true + EPS) - tf.math.log(y_pred + EPS))
    return tf.reduce_sum(w * tf.reduce_mean(kl_elem, axis=0))

def _mae_log(y_true, y_pred):
    EPS = 1e-12
    return tf.reduce_mean(tf.abs(tf.math.log(y_true + EPS) - tf.math.log(y_pred + EPS)))

def composite_loss(lam: float = 0.1, beta: float = 0.0, alpha_entropy: float = 1e-3):
    """
    total = MAE_linear + λ·KL_log  +  α·(-Entropy(y_pred))
    - alpha_entropy ∈ [1e-3, 5e-3] is usually enough to kill the 1e-2 shelf.
    - beta kept for compatibility if you had another term.
    """
    EPS = 1e-12
    def _loss(y_true, y_pred):
        mae_lin = tf.reduce_mean(tf.abs(y_true - y_pred))
        kl_log  = tf.reduce_mean(
            y_true * (tf.math.log(y_true + EPS) - tf.math.log(y_pred + EPS))
        )
        entropy = -tf.reduce_mean(
            tf.reduce_sum(y_pred * tf.math.log(y_pred + EPS), axis=1)
        )
        return mae_lin + lam * kl_log + alpha_entropy * (-entropy)
    return _loss

