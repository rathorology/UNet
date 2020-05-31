import tensorflow as tf
import numpy as np
import glob
import os
from PIL import Image, ImageOps

# @markdown > ReLU slope for `tf.nn.leaky_relu`
relu_alpha = 0.2  # @param {type: "number"}

# @markdown > Dropout rate for `tf.nn.dropout`
dropout_rate = 0.5  # @param {type: "number"}

# @markdown > The padding for the convolution layers.
padding = 'SAME'  # @param [ 'SAME' , 'VALID' ]


def conv2d_down(inputs, filters, stride_size):
    # print( 'conv2d down' )
    out = tf.nn.conv2d(inputs, filters, strides=stride_size, padding=padding)
    return tf.nn.leaky_relu(out, alpha=0.2)


def maxpool_down(inputs, pool_size, stride_size):
    # print( 'maxpool down' )
    return tf.nn.max_pool(inputs, ksize=pool_size, padding='VALID', strides=stride_size)


def conv2d_up(inputs, filters, stride_size, output_shape):
    # print( 'conv2d up' )
    out = tf.nn.conv2d_transpose(inputs, filters, output_shape=output_shape, strides=stride_size, padding=padding)
    return tf.nn.leaky_relu(out, alpha=0.2)


def maxpool_up(inputs, size):
    # print( 'maxpool up' )
    in_dimen = tf.shape(inputs)[1]
    out_dimen = tf.cast(tf.round(in_dimen * size), dtype=tf.int32)
    return tf.image.resize(inputs, [out_dimen, out_dimen], method='nearest')


initializer = tf.initializers.glorot_uniform()


def get_weight(shape, name):
    return tf.Variable(initializer(shape), name=name, trainable=True)


shapes = [
    [3, 3, 3, 16],
    [3, 3, 16, 16],

    [3, 3, 16, 32],
    [3, 3, 32, 32],

    [3, 3, 32, 64],
    [3, 3, 64, 64],

    [3, 3, 64, 128],
    [3, 3, 128, 128],

    [3, 3, 128, 256],
    [3, 3, 256, 256],

    [3, 3, 128, 384],
    [3, 3, 128, 128],

    [3, 3, 64, 192],
    [3, 3, 64, 64],

    [3, 3, 32, 96],
    [3, 3, 32, 32],

    [3, 3, 16, 48],
    [3, 3, 16, 16],

    [1, 1, 16, 1],
]
weights = []
for i in range(len(shapes)):
    weights.append(get_weight(shapes[i], 'weight{}'.format(i)))


def model(x):
    batch_size = tf.shape(x)[0]
    x = tf.cast(x, dtype=tf.float32)
    c1 = conv2d_down(x, weights[0], stride_size=1)
    c1 = conv2d_down(c1, weights[1], stride_size=1)
    p1 = maxpool_down(c1, pool_size=2, stride_size=2)

    c2 = conv2d_down(p1, weights[2], stride_size=1)
    c2 = conv2d_down(c2, weights[3], stride_size=1)
    p2 = maxpool_down(c2, pool_size=2, stride_size=2)

    c3 = conv2d_down(p2, weights[4], stride_size=1)
    c3 = conv2d_down(c3, weights[5], stride_size=1)
    p3 = maxpool_down(c3, pool_size=2, stride_size=2)

    c4 = conv2d_down(p3, weights[6], stride_size=1)
    c4 = conv2d_down(c4, weights[7], stride_size=1)
    p4 = maxpool_down(c4, pool_size=2, stride_size=2)

    c5 = conv2d_down(p4, weights[8], stride_size=1)
    c5 = conv2d_down(c5, weights[9], stride_size=1)

    p5 = maxpool_up(c5, 2)
    concat_1 = tf.concat([p5, c4], axis=-1)
    c6 = conv2d_up(concat_1, weights[10], stride_size=1, output_shape=[batch_size, 16, 16, 128])
    c6 = conv2d_up(c6, weights[11], stride_size=1, output_shape=[batch_size, 16, 16, 128])

    p6 = maxpool_up(c6, 2)
    concat_2 = tf.concat([p6, c3], axis=-1)
    c7 = conv2d_up(concat_2, weights[12], stride_size=1, output_shape=[batch_size, 32, 32, 64])
    c7 = conv2d_up(c7, weights[13], stride_size=1, output_shape=[batch_size, 32, 32, 64])

    p7 = maxpool_up(c7, 2)
    concat_3 = tf.concat([p7, c2], axis=-1)
    c8 = conv2d_up(concat_3, weights[14], stride_size=1, output_shape=[batch_size, 64, 64, 32])
    c8 = conv2d_up(c8, weights[15], stride_size=1, output_shape=[batch_size, 64, 64, 32])

    p8 = maxpool_up(c8, 2)
    concat_4 = tf.concat([p8, c1], axis=-1)
    c9 = conv2d_up(concat_4, weights[16], stride_size=1, output_shape=[batch_size, 128, 128, 16])
    c9 = conv2d_up(c9, weights[17], stride_size=1, output_shape=[batch_size, 128, 128, 16])

    output = tf.nn.conv2d(c9, weights[18], strides=[1, 1, 1, 1], padding=padding)
    outputs = tf.nn.sigmoid(output)
    return outputs


def loss(pred, target):
    return tf.losses.binary_crossentropy(target, pred)


# @markdown > The learning rate used during optimization using Adam.
learning_rate = "0.001"  # @param [ "0.1" , "0.001" , "0.0001" , "0.05" ]
optimizer = tf.optimizers.Adam(learning_rate=float(learning_rate))


def train(model, inputs, outputs):
    with tf.GradientTape() as tape:
        current_loss = loss(model(inputs), outputs)
    grads = tape.gradient(current_loss, weights)
    optimizer.apply_gradients(zip(grads, weights))
    return tf.reduce_mean(current_loss)
