"""
Adaptation of the GRU implementation in keras to allow for auto-regressive generation of sequences.

For the original version of the GRU, please refer to:
https://github.com/fchollet/keras/blob/master/keras/layers/recurrent.py
"""

import keras
import keras.backend as K
import keras.engine as engine
import keras.layers.recurrent as recurrent
import keras.activations as activations
import keras.initializers as initializers
import keras.regularizers as regularizers
import keras.constraints as constraints


class AutoregressiveGRU(recurrent.Recurrent):
    """Autoregressive GRU, feeds the outputs of the previous timestep y_{t-1} as inputs
    for the formation of the current time step's hidden state.

    # Arguments
        units: Positive integer, dimensionality of the hidden space {h_t}.
        output_units: Positive integer, dimensionality of the output space {y_t}.
        output_fn: Callable, takes the hidden state h_t at a given
            time steps and transforms it into an output y_t, which
            is then fed as an input fot the next step (auto-regressive),
            besides the already existing inputs.
        activation: Activation function to use when forming the hidden state
            (see [activations](../activations.md)).
            If you pass None, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        recurrent_activation: Activation function to use for the recurrent step
            (see [activations](../activations.md)).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs.
            (see [initializers](../initializers.md)).
        autoregressive_initializer: Initializer for the `autoregressive kernel`
            weights matrix, used for the linear transformation of the
            previous step outputs.
            (see [initializers](../initializers.md)).
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix,
            used for the linear transformation of the recurrent state.
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizers](../regularizers.md)).
        autoregressive_regularizer: Regularizer function applied to
            the `autoregressive kernel` weights matrix
            (see [regularizers](../regularizers.md)).
        recurrent_regularizer: Regularizer function applied to
            the `recurrent_kernel` weights matrix
            (see [regularizers](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizers](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizers](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).
        autoregressive_constraint: Constraint function applied to the
            `autoregressive kernel` weights matrix
            (see [constraints](../constraints.md)).
        recurrent_constraint: Constraint function applied to
            the `recurrent_kernel` weights matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
        dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the inputs.
        recurrent_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the recurrent state.

    # References
        - [On the Properties of Neural Machine Translation: Encoder-Decoder Approaches](https://arxiv.org/abs/1409.1259)
        - [Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling](http://arxiv.org/abs/1412.3555v1)
        - [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)
    """

    def __init__(self,
                 units,
                 output_units,
                 output_fn=lambda x: x,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 autoregressive_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 autoregressive_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 autoregressive_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 **kwargs):
        super(AutoregressiveGRU, self).__init__(**kwargs)

        self.output_fn = output_fn
        self.units = units
        self.output_units = output_units
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.autoregressive_initializer = initializers.get(autoregressive_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.autoregressive_regularizer = regularizers.get(autoregressive_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.autoregressive_constraint = constraints.get(autoregressive_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        self.state_spec = engine.InputSpec(shape=(None, self.units))

    def get_initial_state(self, inputs):
        # build an all-zero tensor of shape (samples, output_dim)
        initial_state = K.zeros_like(inputs)  # (samples, timesteps, input_dim)
        initial_state = K.sum(initial_state, axis=(1, 2))  # (samples,)
        initial_state = K.expand_dims(initial_state)  # (samples, 1)
        initial_state_h = K.tile(initial_state, [1, self.units])  # (samples, output_dim)
        initial_state_y = K.tile(initial_state, [1, self.output_units])  # (samples, output_dim)
        return [initial_state_h, initial_state_y]

    def build(self, input_shape):

        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        batch_size = input_shape[0] if self.stateful else None
        self.input_dim = input_shape[2]
        self.input_spec[0] = engine.InputSpec(shape=(batch_size, None, self.input_dim))

        if isinstance(self.output_fn, keras.models.Model):
            self.output_fn.build(input_shape=(batch_size, self.units))
            self.trainable_weights += self.output_fn.trainable_weights

            # add regularization losses
            for loss in self.output_fn.losses:
                self.add_loss(loss)

        self.states = [None, None]
        if self.stateful:
            self.reset_states()

        self.kernel = self.add_weight(shape=(self.input_dim, self.units * 3),
                                      name='kernel',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        self.autoregressive_kernel = self.add_weight(shape=(self.output_units, self.units * 3),
                                                     name='autoregressive_kernel',
                                                     initializer=self.autoregressive_initializer,
                                                     regularizer=self.autoregressive_regularizer,
                                                     constraint=self.autoregressive_constraint)

        self.recurrent_kernel = self.add_weight(shape=(self.units, self.units * 3),
                                                name='recurrent_kernel',
                                                initializer=self.recurrent_initializer,
                                                regularizer=self.recurrent_regularizer,
                                                constraint=self.recurrent_constraint)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units * 3,),
                                        name='bias',
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        self.kernel_z = self.kernel[:, :self.units]
        self.autoregressive_kernel_z = self.autoregressive_kernel[:, :self.units]
        self.recurrent_kernel_z = self.recurrent_kernel[:, :self.units]

        self.kernel_r = self.kernel[:, self.units: self.units * 2]
        self.autoregressive_kernel_r = self.autoregressive_kernel[:, self.units:self.units * 2]
        self.recurrent_kernel_r = self.recurrent_kernel[:, self.units: self.units * 2]

        self.kernel_h = self.kernel[:, self.units * 2:]
        self.autoregressive_kernel_h = self.autoregressive_kernel[:, self.units * 2:]
        self.recurrent_kernel_h = self.recurrent_kernel[:, self.units * 2:]

        if self.use_bias:
            self.bias_z = self.bias[:self.units]
            self.bias_r = self.bias[self.units: self.units * 2]
            self.bias_h = self.bias[self.units * 2:]
        else:
            self.bias_z = None
            self.bias_r = None
            self.bias_h = None
        self.built = True

    def preprocess_input(self, inputs, training=None):
        if self.implementation == 0:
            input_shape = K.int_shape(inputs)
            input_dim = input_shape[2]
            timesteps = input_shape[1]

            x_z = recurrent._time_distributed_dense(inputs, self.kernel_z, self.bias_z,
                                                    self.dropout, input_dim, self.units,
                                                    timesteps, training=training)
            x_r = recurrent._time_distributed_dense(inputs, self.kernel_r, self.bias_r,
                                                    self.dropout, input_dim, self.units,
                                                    timesteps, training=training)
            x_h = recurrent._time_distributed_dense(inputs, self.kernel_h, self.bias_h,
                                                    self.dropout, input_dim, self.units,
                                                    timesteps, training=training)
            return K.concatenate([x_z, x_r, x_h], axis=2)
        else:
            return inputs

    def get_constants(self, inputs, training=None):
        constants = []
        if self.implementation != 0 and 0 < self.dropout < 1:
            input_shape = K.int_shape(inputs)
            input_dim = input_shape[-1]
            ones = K.ones_like(K.reshape(inputs[:, 0, 0], (-1, 1)))
            ones = K.tile(ones, (1, int(input_dim)))

            def dropped_inputs():
                return K.dropout(ones, self.dropout)

            dp_mask = [K.in_train_phase(dropped_inputs,
                                        ones,
                                        training=training) for _ in range(3)]
            constants.append(dp_mask)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(3)])

        if 0 < self.recurrent_dropout < 1:
            ones = K.ones_like(K.reshape(inputs[:, 0, 0], (-1, 1)))
            ones = K.tile(ones, (1, self.units))

            def dropped_inputs():
                return K.dropout(ones, self.recurrent_dropout)
            rec_dp_mask = [K.in_train_phase(dropped_inputs,
                                            ones,
                                            training=training) for _ in range(3)]
            constants.append(rec_dp_mask)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(3)])
        return constants

    def step(self, inputs, states):
        h_tm1 = states[0]  # previous memory
        y_tm1 = states[1]  # previous output
        dp_mask = states[2]  # dropout matrices for recurrent units
        rec_dp_mask = states[3]

        y_z = K.dot(y_tm1, self.autoregressive_kernel_z)
        y_r = K.dot(y_tm1, self.autoregressive_kernel_r)
        y_h = K.dot(y_tm1, self.autoregressive_kernel_h)

        if self.implementation == 2:
            matrix_x = K.dot(inputs * dp_mask[0], self.kernel)
            if self.use_bias:
                matrix_x = K.bias_add(matrix_x, self.bias)
            matrix_inner = K.dot(h_tm1 * rec_dp_mask[0],
                                 self.recurrent_kernel[:, :2 * self.units])

            x_z = matrix_x[:, :self.units]
            x_r = matrix_x[:, self.units: 2 * self.units]
            recurrent_z = matrix_inner[:, :self.units]
            recurrent_r = matrix_inner[:, self.units: 2 * self.units]

            z = self.recurrent_activation(x_z + y_z + recurrent_z)
            r = self.recurrent_activation(x_r + y_r + recurrent_r)

            x_h = matrix_x[:, 2 * self.units:]
            recurrent_h = K.dot(r * h_tm1 * rec_dp_mask[0],
                                self.recurrent_kernel[:, 2 * self.units:])
            hh = self.activation(x_h + y_h + recurrent_h)
        else:
            if self.implementation == 0:
                x_z = inputs[:, :self.units]
                x_r = inputs[:, self.units: 2 * self.units]
                x_h = inputs[:, 2 * self.units:]
            elif self.implementation == 1:
                x_z = K.dot(inputs * dp_mask[0], self.kernel_z)
                x_r = K.dot(inputs * dp_mask[1], self.kernel_r)
                x_h = K.dot(inputs * dp_mask[2], self.kernel_h)
                if self.use_bias:
                    x_z = K.bias_add(x_z, self.bias_z)
                    x_r = K.bias_add(x_r, self.bias_r)
                    x_h = K.bias_add(x_h, self.bias_h)
            else:
                raise ValueError('Unknown `implementation` mode.')
            z = self.recurrent_activation(x_z + y_z + K.dot(h_tm1 * rec_dp_mask[0],
                                                            self.recurrent_kernel_z))
            r = self.recurrent_activation(x_r + y_r + K.dot(h_tm1 * rec_dp_mask[1],
                                                            self.recurrent_kernel_r))

            hh = self.activation(x_h + y_h + K.dot(r * h_tm1 * rec_dp_mask[2],
                                                   self.recurrent_kernel_h))
        h = z * h_tm1 + (1 - z) * hh
        if 0 < self.dropout + self.recurrent_dropout:
            h._uses_learning_phase = True

        y = self.output_fn(h)
        return y, [h, y]

    def get_config(self):
        config = {'units': self.units,
                  'activation': activations.serialize(self.activation),
                  'recurrent_activation': activations.serialize(self.recurrent_activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'autoregressive_initializer': initializers.serialize(self.autoregressive_initializer),
                  'recurrent_initializer': initializers.serialize(self.recurrent_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                  'autoregressive_regularizer': regularizers.serialize(self.autoregressive_regularizer),
                  'recurrent_regularizer': regularizers.serialize(self.recurrent_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'activity_regularizer': regularizers.serialize(self.activity_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'autoregressive_constraint': constraints.serialize(self.autoregressive_constraint),
                  'recurrent_constraint': constraints.serialize(self.recurrent_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout}
        base_config = super(AutoregressiveGRU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
