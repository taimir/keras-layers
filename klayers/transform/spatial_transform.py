"""
Spatial transformer layer, compatible with keras APIs
"""
import keras.backend as K
import numpy as np
from keras.layers import Layer
from keras.models import Model


def standardize_coords(coords_grid, dim):
    """
    standardize_coords - standardizes the coordinates in a mesh grid between -1 and 1 for each
    dimension.

    :param coords_grid - shape (dim, width, height, ...)
    :param dim - spatial dimensionality of the data, e.g. 2 for dealing with image data.
        Should match K.int_shape(coords_grid)[0].

    :returns - the standardized coords, shape (dim, width, height, ...)
    """
    maxes = [K.max(coords_grid[i]) for i in range(dim)]
    res = K.stack([2.0 * coords_grid[i] / max_val - 1.0 for i,
                   max_val in zip(range(dim), maxes)], axis=0)
    return res


def affine_transform(coords_grid, params, dim):
    """
    affine_transform represents an affine transformation -
    translation, rotation, scale and skew. Interprets params as the flat parameters of the
    transformation matrix, for each sample.

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
    return transformed


def attention_transform(coords_grid, params, dim):
    """
    attention_transform represents an attention transformation -
    translation and isotropic scaling. Interprets params as the flat parameters of the
    transformation matrix, for each sample.

    :param coords_grid - grid of target coordinates
        shape: (dim, width, height, ...)
    :param params - parametrization of the attention transform
        shape (N, dim + 1), 1 param for the isotropic scaling, dim params for the translation
        component

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
    return transformed


def tps_transform(coords_grid, params):
    """
    tps_transform represents a thin plate spline transformation

    :param coords_grid - grid of target coordinates
    :param params - parametrization of the TPS transformation
    :returns - transformed coords_grid, to be used for sampling from the input image
    """
    raise NotImplementedError


def bitfield(n):
    # http://stackoverflow.com/questions/10321978/integer-to-bitfield-as-a-list
    # a bit faster than int() conversion
    return [1 if digit == '1' else 0 for digit in bin(n)[2:]]


def upscale(coords, maxes):
    coords = K.stack([(coords[:, i] + 1.0) *
                      max_val / 2.0 for i, max_val in enumerate(maxes)], axis=1)
    return coords


def clip(coords, maxes):
    if K.backend() == "tensorflow":
        import tensorflow as tf
        coords = K.stack([tf.clip_by_value(coords[:, i], 0, max_val)
                          for i, max_val in enumerate(maxes)], axis=1)
    else:
        import theano.tensor as T
        coords = K.stack([T.clip(coords[:, i], 0, max_val)
                          for i, max_val in enumerate(maxes)], axis=1)

    coords = K.cast(coords, dtype="int32")
    return coords


def wrap(coords, maxes):
    wrapped_indices = list()
    for i, max_val in enumerate(maxes):
        coords_i = K.cast(coords[:, i], dtype="int32")
        wrapped = coords_i % K.cast(max_val, "int32")
        wrapped_indices.append(wrapped)

    coords = K.stack(wrapped_indices, axis=1)

    return coords


def sample(inputs, coords, dim, wrapped):

    inputs_shape = K.shape(inputs)
    coords_shape = K.shape(coords)
    outputs_shape = K.concatenate([inputs_shape[0:1], coords_shape[2:], inputs_shape[-1:]])

    maxes = [K.cast(inputs_shape[i + 1] - 1, "float32") for i in range(dim)]
    if wrapped:
        coords = wrap(coords, maxes)
    else:
        coords = clip(coords, maxes)

    n = inputs_shape[0]
    n_chan = inputs_shape[-1]

    flat_inputs = K.reshape(inputs, (-1, n_chan))

    flat_coords = K.flatten(coords[:, -1])
    for i in reversed(range(dim - 1)):
        flat_coords += K.prod(inputs_shape[1:i + 2]) * K.flatten(coords[:, i])

    coords_per_sample = K.prod(coords_shape[2:])

    # add the offsets for each sample in the minibatch
    if K.backend() == "tensorflow":
        import tensorflow as tf
        offsets = tf.range(n) * K.prod(inputs_shape[1:-1])
    else:
        import theano.tensor as T
        offsets = T.arange(n) * K.prod(inputs_shape[1:-1])

    offsets = K.reshape(offsets, (-1, 1))
    offsets = K.tile(offsets, (1, coords_per_sample))
    offsets = K.flatten(offsets)
    flat_coords += offsets

    outputs = K.gather(flat_inputs, flat_coords)
    outputs = K.reshape(outputs, outputs_shape)
    return outputs


def interpolate_bilinear(coords, inputs, dim, wrap=False):
    """
    interpolate_bilinear - the default interpolation kernel to be used with the spatial
    transformer. Differential w.r.t. both the indices and the input tensors to be sampled.

    :param coords
        shape: (N, dim, width, height, ...)
    :param inputs
        shape: (N, width, height, .. n_chan)
    :param dim - dimensionality of the data, e.g. 2 if inputs is a batch of images
    :param wrap - whether to wrap, or otherwise clip during the interpolation

    :returns - the sampled result
        :shape (N, width, height, ..., n_chan), where width, height, ... come from the
        coords shape
    """

    coords_float = coords

    inputs_shape = K.shape(inputs)
    maxes = [K.cast(inputs_shape[i + 1] - 1, "float32") for i in range(dim)]
    coords = upscale(coords, maxes)

    # floored coordinates, time to build the surrounding points based on them
    if K.backend() == "tensorflow":
        import tensorflow as tf
        coords = tf.floor(coords)
    else:
        import theano.tensor as T
        coords = T.floor(coords)

    # construct the surrounding 2^dim coord sets which will all be used for interpolation
    # (e.g. corresponding to the 4 points in 2D that surround the point to be interpolated,
    # or to the 8 points in 3D, etc ...)
    surround_coord_sets = []
    for i in range(2 ** dim):
        offsets = K.variable(np.array(bitfield(i)))
        offsets = K.reshape(offsets, shape=[1, -1] + [1] * dim)
        surround_coord_set = coords + offsets
        surround_coord_sets.append(surround_coord_set)

    # sample for each of the surrounding points before interpolating
    surround_inputs = []
    for coords_set in surround_coord_sets:
        surround_input = sample(inputs, coords_set, dim, wrapped=wrap)
        surround_inputs.append(surround_input)

    # Bilinear interpolation, this part of the kernel lets the gradients flow through the
    # coords as well as the inputs
    products = list()
    for coords_set, surround_input in zip(surround_coord_sets, surround_inputs):
        if K.backend() == "tensorflow":
            import tensorflow as tf
            # shape N, width, height, ...
            product = tf.reduce_prod(1 - tf.abs(coords_set - coords_float), axis=1)
        else:
            import theano.tensor as T
            product = T.prod(1 - T.abs(coords_set - coords_float), axis=1)

        # shape: (N, width, height, ..., n_channels)
        product = surround_input * K.expand_dims(product, -1)
        products.append(product)

    return sum(products)


def interpolate_nearest(coords, inputs, dim, wrap=False):
    """
    CAUTION: This interpolation kernel is not differentiable. Use only if you do not need
    gradients flowing back throught he localization network.

    interpolate_nearest - samples with coords from inputs, interpolating the results
    via nearest neighbours rounding of the indices (which are not whole numbers yet).

    :param coords
        shape: (N, dim, width, height, ...)
    :param inputs
        shape: (N, width, height, .. n_chan)
    :param dim - dimensionality of the data, e.g. 2 if inputs is a batch of images
    :param wrap - whether to wrap, or otherwise clip during the interpolation

    :returns - the sampled result
        :shape (N, width, height, ..., n_chan), where width, height, ... come from the
        coords shape
    """
    inputs_shape = K.shape(inputs)

    maxes = [K.cast(inputs_shape[i + 1] - 1, "float32") for i in range(dim)]

    coords = upscale(coords, maxes)
    coords = K.round(coords)

    return sample(inputs, coords, dim, wrapped=wrap)


class SpatialTransform(Layer):
    """
    SpatialTransformer layer, which can automatically predict the parameters of a spatial
    transformation that is then applied to the input.

    :param output_shape - desired shape of the output image / volume / ..., without the channels
        e.g. (width, height) or (width, height, depth) for 3D volumes
    :param loc_network - neural network that will produce the transformation parameters
    :param grid_transform_fn - function that interprets the parameters in the output of
        loc_network as a transformation of image coordinates and applies it
    :param interpolation_fn - function that samples the image with interpolation
    :param wrap - whether to wrap, or otherwise clip during the interpolation
    :param **kwargs
    """

    def __init__(self, output_grid_shape,
                 loc_network,
                 grid_transform_fn,
                 interpolation_fn=interpolate_bilinear,
                 wrap=False,
                 **kwargs):
        self.output_grid_shape = output_grid_shape
        self.loc_network = loc_network
        self.grid_transform_fn = grid_transform_fn
        self.interpolation_fn = interpolation_fn
        self.wrap = wrap

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
        transformed_image = self.interpolation_fn(transformed_coords, x, dim=dim, wrap=self.wrap)
        return transformed_image

    def compute_output_shape(self, input_shape):
        # add the channels dimension
        return (input_shape[0],) + self.output_grid_shape + (input_shape[-1],)
