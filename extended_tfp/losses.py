import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

def negative_log_likelihood(y_true, y_pred_rv):
    '''
    stole from https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/jupyter_notebooks/Probabilistic_Layers_Regression.ipynb
    '''
    return -y_pred_rv.log_prob(y_true)
