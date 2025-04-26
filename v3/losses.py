# losses.py ----------------------------------------------------------
import tensorflow as tf
EPS = 1e-12          # using 1e-12 everywhere avoids log(0)

def composite_loss(lambda_=0.7):
    kl = tf.keras.losses.KLDivergence()
    log10 = tf.math.log(10.0)

    def mae_log(y_true, y_pred):
        y_true_log = tf.math.log(y_true + EPS) / log10
        y_pred_log = tf.math.log(y_pred + EPS) / log10
        return tf.reduce_mean(tf.abs(y_true_log - y_pred_log))

    def _loss(y_true, y_pred):
        return lambda_ * kl(y_true, y_pred) + (1.0 - lambda_) * mae_log(y_true, y_pred)

    return _loss
