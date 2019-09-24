import tensorflow as tf

# TODO make it pretty..

def compute_variation_ratio(prob_samples):
    '''
    Referenced 3.3.1 of following thesis.
        Y. Gal, "Uncertainty in Deep Learning", University of Cambridge (2016)

    Args:
      prob_samples: [num_samples, batch_size, num_classes]
    Returns:
      variation_ratio: [batch_size]
    '''
    num_samples = len(prob_samples)
    mode = tf.reduce_max(prob_samples, axis=2, keepdims=True)
    is_mode = tf.dtypes.cast(prob_samples == mode, prob_samples.dtype)
    num_mode = tf.reduce_sum(is_mode, axis=2, keepdims=True)
    mode_frequency = tf.reduce_max(tf.reduce_sum(is_mode / num_mode, axis=0), axis=1)
    return 1 - (mode_frequency / num_samples)


def compute_predictive_entropy(prob_samples, eps=1e-12):
    '''
    Referenced 3.3.1 of following thesis.
        Y. Gal, "Uncertainty in Deep Learning", University of Cambridge (2016)

    Args:
      prob_samples: [num_samples, batch_size, num_classes]
    Returns:
      variation_ratio: [batch_size]
    '''
    # predictive probability
    pred_prob = tf.reduce_mean(prob_samples, axis=0)
    pred_log_prob = tf.math.log(tf.clip_by_value(pred_prob, eps, 1))
    return -tf.reduce_sum(pred_prob * pred_log_prob, axis=1)


def compute_mutual_information(prob_samples, eps=1e-12):
    '''
    Referenced 3.3.1 of following thesis.
        Y. Gal, "Uncertainty in Deep Learning", University of Cambridge (2016)

    Args:
      prob_samples: [num_samples, batch_size, num_classes]
    Returns:
      variation_ratio: [batch_size]
    '''
    predictive_entropy = compute_predictive_entropy(prob_samples)

    log_prob_samples = tf.math.log(tf.clip_by_value(prob_samples, eps, 1))
    neg_entropy_samples = tf.reduce_sum(prob_samples * log_prob_samples, axis=2)
    return predictive_entropy + tf.reduce_mean(neg_entropy_samples, axis=0)
