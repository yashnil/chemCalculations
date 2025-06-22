
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
    y_true_log = tf.math.log(y_true + EPS) / LOG10
    y_pred_log = tf.math.log(y_pred + EPS) / LOG10
    return tf.reduce_mean(tf.abs(y_true_log - y_pred_log))

def composite_loss(lam: float = 0.6,
                   A_mat: tf.Tensor = None,
                   b_vec: tf.Tensor = None,
                   beta: float = 0.0):
    """
    If you pass A_mat (shape [n_spec→n_elem]) and b_vec ([n_elem]),
    a small mass-balance penalty β·∥A·y_pred - b∥² will be added.
    """
    def _loss(y_true, y_pred):
        # 1) balanced‐KL per sample
        w = 1.0 / (tf.reduce_mean(y_true, axis=0) + EPS)
        w = w / tf.reduce_sum(w)
        kl_elem = y_true * (tf.math.log(y_true + EPS) - tf.math.log(y_pred + EPS))
        kl_per_sample = tf.reduce_sum(w * kl_elem, axis=1)    # [batch]

        # 2) log‐MAE per sample
        y_true_log = tf.math.log(y_true + EPS) / LOG10
        y_pred_log = tf.math.log(y_pred + EPS) / LOG10
        mae_log_per_sample = tf.reduce_mean(tf.abs(y_true_log - y_pred_log), axis=1)

        loss = lam * kl_per_sample + (1.0 - lam) * mae_log_per_sample  # [batch]

        # 3) optional mass‐balance
        if A_mat is not None and b_vec is not None and beta > 0.0:
            # y_pred [batch,n_spec] → A_mat [n_spec,n_elem] → [batch,n_elem]
            mb = tf.reduce_mean((tf.matmul(y_pred, A_mat) - b_vec)**2, axis=1)
            loss = loss + beta * mb

        return loss
    return _loss
