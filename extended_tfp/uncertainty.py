import tensorflow as tf


def compute_entropy(probability, axis, epsilon=1e-7):
    '''
    Args:
      probability:
      axis:
      epsilon:
    Returns:
      entropy
    '''
    is_zero = probability == 0
    has_zero = tf.math.reduce_any(is_zero)
    if has_zero:
        mask = epsilon * tf.ones_like(probability)
        masked_probability = tf.where(is_zero, mask, probability)
        log_probability = tf.math.log(masked_probability)
    else:
        log_probability = tf.math.log(probability)

    # entropy
    entropy = -tf.reduce_sum(probability * log_probability, axis=axis)
    return entropy


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

    max_probs = tf.reduce_max(prob_samples, axis=2, keepdims=True)
    is_mode = tf.dtypes.cast(prob_samples == max_probs, prob_samples.dtype)

    # for multi-modal
    num_mode = tf.reduce_sum(is_mode, axis=2, keepdims=True)
    is_mode = is_mode / num_mode

    frequency = tf.reduce_sum(is_mode, axis=0)
    mode_frequency = tf.reduce_max(frequency, axis=1)
    return 1 - (mode_frequency / num_samples)


def compute_predictive_entropy(prob_samples, eps=1e-12):
    '''
    Referenced 3.3.1 of following thesis.
        Y. Gal, "Uncertainty in Deep Learning", University of Cambridge (2016)

    Args:
      prob_samples: [num_samples, batch_size, num_classes]
    Returns:
      predictive_entropy: [batch_size]
    '''
    pred_prob = tf.reduce_mean(prob_samples, axis=0)
    return compute_entropy(pred_prob, axis=1)


def compute_expected_entropy(prob_samples):
    '''
    Args:
      prob_samples: [num_samples, batch_size, num_classes]
    '''
    entropy_samples = compute_entropy(prob_samples, axis=2)
    return tf.reduce_mean(entropy_samples, axis=0)


def compute_mutual_information(prob_samples, eps=1e-12):
    '''
    Referenced 3.3.1 of following thesis.
        Y. Gal, "Uncertainty in Deep Learning", University of Cambridge (2016)

    Args:
      prob_samples: [num_samples, batch_size, num_classes]
    Returns:
      mutual_information: [batch_size, ]
    '''
    predictive_entropy = compute_predictive_entropy(prob_samples)
    expected_entropy = compute_expected_entropy(prob_samples)
    return predictive_entropy - expected_entropy
