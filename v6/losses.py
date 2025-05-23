
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


# ------------------------------------------------------------------
# helper: KL divergence re-weighted by the inverse mean abundance
# ------------------------------------------------------------------
def _balanced_kl(y_true, y_pred):
    """
    KL divergence where each species is weighted by
        w_i = 1 / ⟨y_true_i⟩    (normalised so Σw = 1)

    This prevents the gradient from being dominated by the
    handful of abundant species and pulls the rare end of the
    distribution (≈ 1e-10 … 1e-6) toward the 1:1 line.
    """
    # average abundance per species over the batch
    w = 1.0 / (tf.reduce_mean(y_true, axis=0) + EPS)
    w = w / tf.reduce_sum(w)

    kl_elem = y_true * (
        tf.math.log(y_true + EPS) - tf.math.log(y_pred + EPS)
    )                                                   

    # mean over batch, then weighted sum over species → scalar
    return tf.reduce_sum(w * tf.reduce_mean(kl_elem, axis=0))


# ------------------------------------------------------------------
# helper: MAE in log10 space
# ------------------------------------------------------------------
def _mae_log(y_true, y_pred):
    y_true_log = tf.math.log(y_true + EPS) / LOG10
    y_pred_log = tf.math.log(y_pred + EPS) / LOG10
    return tf.reduce_mean(tf.abs(y_true_log - y_pred_log))


# ------------------------------------------------------------------
# public factory  : composite loss = λ·balanced_KL + (1-λ)·log-MAE
# ------------------------------------------------------------------
def composite_loss(lam: float = 0.6):
    def _loss(y_true, y_pred):
        # y_true, y_pred: [batch, species]
        # 1) compute the inverse–mean weights once per batch
        w = 1.0 / (tf.reduce_mean(y_true, axis=0) + EPS)  
        w = w / tf.reduce_sum(w)                        # [species]

        # 2) per-sample KL divergence: sum_i w_i * (y_true*log(y_true/y_pred))_i
        kl_elem      = y_true * (tf.math.log(y_true+EPS) - tf.math.log(y_pred+EPS))
        kl_per_sample = tf.reduce_sum(w * tf.reduce_mean(kl_elem, axis=0), axis=0)
        # Actually we need per-sample → compute mean over species per sample then weight
        kl_per_sample = tf.reduce_sum(w * tf.reduce_mean(kl_elem, axis=0), axis=0)
        # (that’s still scalar—so better do species_weight * per-sample kl)

        # Instead, do it this way:
        kl_per_sample = tf.reduce_sum(w * kl_elem, axis=1)  # [batch]

        # 3) per-sample log-MAE
        y_true_log = tf.math.log(y_true + EPS) / LOG10
        y_pred_log = tf.math.log(y_pred + EPS) / LOG10
        mae_log_per_sample = tf.reduce_mean(tf.abs(y_true_log - y_pred_log), axis=1)  # [batch]

        # 4) combine
        return lam * kl_per_sample + (1.0 - lam) * mae_log_per_sample  # [batch]
    return _loss
