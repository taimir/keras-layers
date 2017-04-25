"""
Spatial transformer layer, compatible with keras APIs
"""
import keras.backend as K
import numpy as np
from keras.layers import Layer
from keras.models import Model


def standardize_coords(coords_grid, dim):
    maxes = [K.max(coords_grid[i]) for i in range(dim)]
    res = K.stack([2.0 * coords_grid[i] / max_val - 1.0 for i,
                   max_val in zip(range(dim), maxes)], axis=0)
    return res


def affine_transform(coords_grid, params, dim):
    """
    affine_transform represents an affine transformation -
    translation, rotation, scale and skew

    :param coords_grid - grid of target coordinates
        shape: (dim, width, height, ...)
    :param params - parametrization of the affine transform
        shape: (N, dim^2 + dim), dim^2 params for the rotation matrix + dim params
        for the translation component
    :returns - transformed coords_grid, to be used for sampling from the input image
        shape: (N, dim, width, height, ...)
    """
    # standardize, extend to homogenous coordinates
    coords_grid = standardize_coords(coords_grid, dim)
    ones_pad = K.expand_dims(K.ones_like(coords_grid[0]), axis=0)
    coords_grid = K.concatenate([coords_grid, ones_pad], axis=0)

    # interpret the params as an affine transform matrix
    transform_mat = K.reshape(x=params, shape=(-1, dim, dim + 1))

    # apply the transformation (keras tensor product uses axis -2 for the second tensor)
    coords_grid = K.permute_dimensions(x=coords_grid, pattern=(1, 0, 2))
    transformed = K.dot(transform_mat, coords_grid)
    clipped = K.clip(transformed, min_value=-1, max_value=1)
    return clipped


def attention_transform(coords_grid, params, dim):
    """
    attention_transform represents an attention transformation -
    translation and isotropic scaling

    :param coords_grid - grid of target coordinates
    :param params - parametrization of the attention transform, shape (N, dim + 1)
       1 param for the isotropic scaling, dim params for the translation component
    :returns - transformed coords_grid, to be used for sampling from the input image
    """
    # standardize, extend to homogenous coordinates
    coords_grid = standardize_coords(coords_grid, dim)
    ones_pad = K.expand_dims(K.ones_like(coords_grid[0]), axis=0)
    coords_grid = K.concatenate([coords_grid, ones_pad], axis=0)

    # form the attention matrix, one part is a lambda * I, the other is the translation
    n = K.shape(params)[0]
    # scaling part: repeat lambda * I (for each param row)
    # shape: (N, dim, dim)
    scale_part = params[:, 0:1] * K.tile(x=K.reshape(K.eye(dim), shape=(1, -1)), n=[n, 1])
    scale_part = K.reshape(scale_part, shape=[n, dim, dim])

    # translation part
    t_part = K.reshape(params[:, 1:], shape=(-1, dim, 1))
    # attention matrix
    transform_mat = K.concatenate([scale_part, t_part], axis=-1)

    # apply the transformation (keras tensor product uses axis -2 for the second tensor)
    coords_grid = K.permute_dimensions(x=coords_grid, pattern=(1, 0, 2))
    transformed = K.dot(transform_mat, coords_grid)
    clipped = K.clip(transformed, min_value=-1, max_value=1)
    return clipped


def tps_transform(coords_grid, params):
    """
    tps_transform represents a thin plate spline transformation

    :param coords_grid - grid of target coordinates
    :param params - parametrization of the TPS transformation
    :returns - transformed coords_grid, to be used for sampling from the input image
    """
    raise NotImplementedError


def bilinear_interpolate(selection_indices, inputs, dim):
    raise NotImplementedError


def interpolate_nearest(selection_indices, inputs, dim):
    """interpolate_nearest

    :param selection_indices - (N, dim, width, height, ...)
    :param inputs - (N, width, height, .. n_chan)
    :param dim
    """

    inputs_shape = K.shape(inputs)
    indices_shape = K.shape(selection_indices)
    outputs_shape = K.concatenate([inputs_shape[0:1], indices_shape[2:], inputs_shape[-1:]])

    n = inputs_shape[0]
    n_chan = inputs_shape[-1]

    maxes = [K.cast(inputs_shape[i + 1] - 1, "float32") for i in range(dim)]
    selection_indices = K.stack([(selection_indices[:, i] + 1.0) *
                                 max_val / 2.0 for i, max_val in zip(range(dim), maxes)], axis=1)

    flat_inputs = K.reshape(inputs, (-1, n_chan))
    selection_indices = K.cast(K.round(selection_indices), dtype="int32")

    flat_indices = K.flatten(selection_indices[:, -1])
    for i in reversed(range(dim - 1)):
        flat_indices += K.prod(inputs_shape[1:i + 2]) * K.flatten(selection_indices[:, i])

    indices_per_sample = K.prod(indices_shape[2:])

    # add the offsets for each sample in the minibatch
    if K.backend() == "tensorflow":
        import tensorflow as tf
        offsets = tf.range(n) * indices_per_sample
    else:
        import theano.tensor as T
        offsets = T.arange(n) * indices_per_sample
    offsets = K.reshape(offsets, (-1, 1))
    offsets = K.tile(offsets, (1, indices_per_sample))
    offsets = K.flatten(offsets)
    flat_indices += offsets

    outputs = K.gather(flat_inputs, flat_indices)
    outputs = K.reshape(outputs, outputs_shape)
    return outputs


class SpatialTransform(Layer):

    def __init__(self, output_grid_shape,
                 loc_network,
                 grid_transform_fn,
                 interpolation_fn,
                 **kwargs):
        """__init__

        :param output_shape - desired shape of the output image / volume, without the channels
            e.g. (width, height) or (width, height, depth) for 3D volumes
        :param loc_network - neural network that will produce the transformation parameters
        :param grid_transform_fn - function that interprets the parameters in the output of
            loc_network as a transformation of image coordinates and applies it
        :param interpolation_fn - function that samples the image with interpolation
        :param **kwargs
        """
        self.output_grid_shape = output_grid_shape
        self.loc_network = loc_network
        self.grid_transform_fn = grid_transform_fn
        self.interpolation_fn = interpolation_fn

        # initialize the coords grid
        indices = np.indices(self.output_grid_shape, dtype="float32")
        self.coords_grid = K.variable(indices, name="grid_indices")
        super(SpatialTransform, self).__init__(**kwargs)

    def build(self, input_shape):
        if isinstance(self.loc_network, Layer) or isinstance(self.loc_network, Model):
            if hasattr(self, 'previous'):
                self.loc_network.set_previous(self.previous)
            self.loc_network.build()
            self.trainable_weights = self.loc_network.trainable_weights
            self.regularizers = self.loc_network.regularizers
            self.constraints = self.loc_network.constraints
            self.input = self.loc_network.input
        super(SpatialTransform, self).build(input_shape)

    def call(self, x):
        params = self.loc_network(x)
        # dimensionality of the data is all dims without batch_size and channels
        dim = len(K.int_shape(x)) - 2
        transformed_coords = self.grid_transform_fn(coords_grid=self.coords_grid,
                                                    params=params,
                                                    dim=dim)
        transformed_image = self.interpolation_fn(transformed_coords, x, dim=dim)
        return transformed_image

    def compute_output_shape(self, input_shape):
        # add the channels dimension
        return (input_shape[0],) + self.output_grid_shape + (input_shape[-1],)


def rot_scale_matrix(angle, scale, trans):
    import numpy as np
    rot_mat = np.array([[np.cos(angle), -np.sin(angle)],
                        [np.sin(angle), np.cos(angle)]],
                       dtype="float32")
    scale_mat = np.array([[1/scale, 0.0],
                          [0.0, 1/scale]],
                         dtype="float32")
    mat = np.dot(scale_mat, rot_mat)
    trans = trans[:, np.newaxis]
    hom_mat = np.concatenate([mat, trans], axis=1)
    return np.reshape(hom_mat, [1, -1])


def test_affine_transform():
    import numpy as np
    from keras.datasets import mnist
    from keras.layers import Input

    (x_train, _), (_, _) = mnist.load_data()

    samples = x_train[:10, :, :, np.newaxis]

    def loc_network(x):
        import tensorflow as tf
        flat_mat = rot_scale_matrix(-1.5, 0.5, np.array([0.3, -0.1]))
        return tf.convert_to_tensor(np.tile(flat_mat, [10, 1]), dtype="float32")

    inputs = Input(shape=[28, 28, 1])
    st = SpatialTransform(output_grid_shape=(56, 56),
                          loc_network=loc_network,
                          grid_transform_fn=affine_transform,
                          interpolation_fn=interpolate_nearest)
    outputs = st(inputs)
    sess = K.get_session()

    res = sess.run(outputs, feed_dict={inputs: samples})

    return samples, res


def test_attention_transform():
    import numpy as np
    from keras.datasets import mnist
    from keras.layers import Input

    (x_train, _), (_, _) = mnist.load_data()

    samples = x_train[:10, :, :, np.newaxis]

    def loc_network(x):
        import tensorflow as tf
        flat_mat = np.array([2.0, 0.3, -0.1])
        flat_mat = flat_mat[np.newaxis, :]
        return tf.convert_to_tensor(np.tile(flat_mat, [10, 1]), dtype="float32")

    inputs = Input(shape=[28, 28, 1])
    st = SpatialTransform(output_grid_shape=(56, 56),
                          loc_network=loc_network,
                          grid_transform_fn=attention_transform,
                          interpolation_fn=interpolate_nearest)
    outputs = st(inputs)
    sess = K.get_session()

    res = sess.run(outputs, feed_dict={inputs: samples})

    return samples, res


if __name__ == "__main__":
    # original, transformed = test_affine_transform()
    original, transformed = test_attention_transform()
    import matplotlib.pyplot as plt
    plt.figure(1)
    plt.subplot(211)
    plt.imshow(original[0, :, :, 0], cmap="gray", interpolation="none")
    plt.subplot(212)
    plt.imshow(transformed[0, :, :, 0], cmap="gray", interpolation="none")
    plt.show()
