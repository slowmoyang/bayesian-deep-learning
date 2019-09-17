from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.ops import nn_ops
from tensorflow_probability.python.layers.dense_variational_v2 import _make_kl_divergence_penalty


class ConvVariational(tf.keras.layers.Layer):
    def __init__(self, rank,
                 filters,
                 kernel_size,
                 make_posterior_fn,
                 make_prior_fn,
                 kl_weight=None,
                 kl_use_exact=False,
                 strides=1,
                 padding='valid',
                 data_format=None,
                 dilation_rate=1,
                 activation=None,
                 use_bias=True,
                 activity_regularizer=None,
                 **kwargs):
        super(ConvVariational, self).__init__(
            activity_regularizer=tf.keras.regularizers.get(activity_regularizer),
            **kwargs)
        self.rank = rank

        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(
            kernel_size, rank, 'kernel_size')

        self._make_posterior_fn = make_posterior_fn
        self._make_prior_fn = make_prior_fn
        self._kl_divergence_fn = _make_kl_divergence_penalty(
            kl_use_exact, weight=kl_weight)


        self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        if (self.padding == 'causal' and not isinstance(self,
                                                        (Conv1DVariational))):
            raise ValueError('Causal padding is only supported for `Conv1DVariational`.')
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(
            dilation_rate, rank, 'dilation_rate')
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias

    def build(self, input_shape):
        dtype = tf.as_dtype(self.dtype or tf.keras.backend.floatx())
        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError('Unable to build `Dense` layer with non-floating point dtype {:s}'.format(dtype))


        input_shape = tf.TensorShape(input_shape)
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        input_dim = tf.compat.dimension_value(input_shape[channel_axis])
        if input_dim is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        self.kernel_shape = self.kernel_size + (input_dim, self.filters)

        dtype = tf.as_dtype(self.dtype or tf.keras.backend.floatx())

        self.input_spec = tf.keras.layers.InputSpec(
            ndim=self.rank + 2, axes={channel_axis: input_dim})

        self._convolution_op = nn_ops.Convolution(
            input_shape,
            filter_shape=tf.TensorShape(self.kernel_shape),
            dilation_rate=self.dilation_rate,
            strides=self.strides,
            padding=self.padding.upper(),
            data_format=conv_utils.convert_data_format(
                self.data_format, self.rank + 2))

        self.num_weights = np.prod(self.kernel_shape)

        self._posterior = self._make_posterior_fn(
            self.num_weights,
            self.filters if self.use_bias else 0,
            dtype)

        self._prior = self._make_prior_fn(
            self.num_weights,
            self.filters if self.use_bias else 0,
            dtype)

        self.built = True

    def call(self, inputs):
        dtype = tf.as_dtype(self.dtype or tf.keras.backend.floatx())
        inputs = tf.cast(inputs, dtype, name='inputs')

        q = self._posterior(inputs)
        r = self._prior(inputs)
        self.add_loss(self._kl_divergence_fn(q, r))

        w = tf.convert_to_tensor(value=q)
        if self.use_bias:
            split_sizes = [self.num_weights, self.filters]
            kernel, bias = tf.split(w, split_sizes, axis=-1)
        else:
            kernel, bias = w, None

        kernel = tf.reshape(kernel, shape=tf.concat([
            tf.shape(input=kernel)[:-1],
            self.kernel_shape,
        ], axis=0))

        outputs = self._convolution_op(inputs, kernel)

        if self.use_bias:
            if self.data_format == 'channels_first':
                if self.rank == 1:
                    # nn.bias_add does not accept a 1D input tensor.
                    bias = tf.reshape(bias, (1, self.filters, 1))
                    outputs += bias
                else:
                    outputs = tf.nn.bias_add(outputs, bias, data_format='NCHW')
            else:
                outputs = tf.nn.bias_add(outputs, bias, data_format='NHWC')

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def _compute_causal_padding(self):
        left_pad = self.dilation_rate[0] * (self.kernel_size[0] - 1)
        if self.data_format == 'channels_last':
            causal_padding = [[0, 0], [left_pad, 0], [0, 0]]
        else:
            causal_padding = [[0, 0], [0, 0], [left_pad, 0]]
        return causal_padding


class Conv1DVariational(ConvVariational):
    def __init__(self,
                 filters,
                 kernel_size,
                 make_posterior_fn,
                 make_prior_fn,
                 kl_weight=None,
                 kl_use_exact=False,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation=None,
                 use_bias=True,
                 activity_regularizer=None,
                 **kwargs):
        super(Conv1DVariational, self).__init__(
            rank=1,
            filters=filters,
            kernel_size=kernel_size,
            make_posterior_fn=make_posterior_fn,
            make_prior_fn=make_prior_fn,
            kl_weight=kl_weight,
            kl_use_exact=kl_use_exact,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            activity_regularizer=activity_regularizer,
            **kwargs)

    def call(self, inputs):
        if self.padding == 'causal':
            inputs = array_ops.pad(inputs, self._compute_causal_padding())
        return super(Conv1D, self).call(inputs)


class Conv2DVariational(ConvVariational):
    def __init__(self,
                 filters,
                 kernel_size,
                 make_posterior_fn,
                 make_prior_fn,
                 kl_weight=None,
                 kl_use_exact=False,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation=None,
                 use_bias=True,
                 activity_regularizer=None,
                 **kwargs):
        super(Conv2DVariational, self).__init__(
            rank=2,
            filters=filters,
            kernel_size=kernel_size,
            make_posterior_fn=make_posterior_fn,
            make_prior_fn=make_prior_fn,
            kl_weight=kl_weight,
            kl_use_exact=kl_use_exact,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            activity_regularizer=activity_regularizer,
            **kwargs)
