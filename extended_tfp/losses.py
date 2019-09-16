import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

def negative_log_likelihood(y_true, y_pred_rv):
    return -y_pred_rv.log_prob(y_true)
