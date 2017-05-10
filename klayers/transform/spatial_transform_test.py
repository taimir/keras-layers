import keras.backend as K
from klayers.transform import affine_transform, attention_transform, interpolate_gaussian
from klayers.transform import SpatialTransform


def rot_scale_matrix(angle, scale, trans):
    import numpy as np
    rot_mat = np.array([[np.cos(angle), -np.sin(angle)],
                        [np.sin(angle), np.cos(angle)]],
                       dtype="float32")
    scale_mat = np.array([[scale, 0.0],
                          [0.0, scale]],
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
        flat_mat = rot_scale_matrix(0, 1.0, np.array([2.7, -0.0]))
        return tf.convert_to_tensor(np.tile(flat_mat, [10, 1]), dtype="float32")

    def interpolate_fn(coords, inputs, dim, wrap):
        return interpolate_gaussian(coords, inputs, dim, wrap=wrap,
                                    kernel_size=5, stddev=1.0)

    inputs = Input(shape=[28, 28, 1])
    st = SpatialTransform(output_grid_shape=(44, 44),
                          loc_network=loc_network,
                          grid_transform_fn=affine_transform,
                          interpolation_fn=interpolate_fn,
                          wrap=True)
    outputs = st(inputs)
    sess = K.get_session()

    res = sess.run(outputs, feed_dict={inputs: samples})

    return samples, res


def test_attention_transform():
    """test_attention_transform"""
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
                          grid_transform_fn=attention_transform)
    outputs = st(inputs)
    sess = K.get_session()

    res = sess.run(outputs, feed_dict={inputs: samples})

    return samples, res


if __name__ == "__main__":
    original, transformed = test_affine_transform()
    # original, transformed = test_attention_transform()
    import matplotlib.pyplot as plt
    for i in range(3):
        plt.figure(1)
        plt.subplot(211)
        plt.imshow(original[i, :, :, 0], cmap="gray", interpolation="none")
        plt.subplot(212)
        plt.imshow(transformed[i, :, :, 0], cmap="gray", interpolation="none")
        plt.show()
