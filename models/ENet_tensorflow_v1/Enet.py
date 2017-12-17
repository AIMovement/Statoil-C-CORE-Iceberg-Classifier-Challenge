"""Enet implementation in Tensorflow."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf
from tensorflow.python.ops.image_ops_impl import ResizeMethod
#slim = tf.contrib.slim


def unpool(inputs, pooling_indices, output_shape=None, k_size=[1, 2, 2, 1]):
    # NOTE! this function is based on the implementation by kwotsin in
    # https://github.com/kwotsin/TensorFlow-ENet

    # inputs has shape [batch_size, height, width, channels]

    # pooling_indices: pooling indices of the previously max_pooled layer

    # output_shape: what shape the returned tensor should have

    pooling_indices = tf.cast(pooling_indices, tf.int32)
    input_shape = tf.shape(inputs, out_type=tf.int32)

    one_like_pooling_indices = tf.ones_like(pooling_indices, dtype=tf.int32)
    batch_shape = tf.concat([[input_shape[0]], [1], [1], [1]], 0)
    batch_range = tf.reshape(tf.range(input_shape[0], dtype=tf.int32), shape=batch_shape)
    b = one_like_pooling_indices*batch_range
    y = pooling_indices//(output_shape[2]*output_shape[3])
    x = (pooling_indices//output_shape[3]) % output_shape[2]
    feature_range = tf.range(output_shape[3], dtype=tf.int32)
    f = one_like_pooling_indices*feature_range

    inputs_size = tf.size(inputs)
    indices = tf.transpose(tf.reshape(tf.stack([b, y, x, f]), [4, inputs_size]))
    values = tf.reshape(inputs, [inputs_size])
    if output_shape[0] == None:
        output_shape[0] = 1
    ret = tf.scatter_nd(indices, values, output_shape)

    return ret



def spatial_dropout(x, keep_prob, is_training = True, seed=1234):
    """Spatial dropout."""
    if is_training:
        # x is a convnet activation with shape BxWxHxF where F is the
        # number of feature maps for that layer
        # keep_prob is the proportion of feature maps we want to keep

        # get the batch size and number of feature maps
        num_feature_maps = [tf.shape(x)[0], tf.shape(x)[3]]

        # get some uniform noise between keep_prob and 1 + keep_prob
        random_tensor = keep_prob
        random_tensor += tf.random_uniform(num_feature_maps,
                                           seed=seed,
                                           dtype=x.dtype)

        # if we take the floor of this, we get a binary matrix where
        # (1-keep_prob)% of the values are 0 and the rest are 1
        binary_tensor = tf.floor(random_tensor)

        # Reshape to multiply our feature maps by this tensor correctly
        binary_tensor = tf.reshape(binary_tensor,
                                   [tf.shape(x)[0], 1, 1, tf.shape(x)[3]])
        # Zero out feature maps where appropriate; scale up to compensate
        ret = tf.div(x, keep_prob) * binary_tensor
    else:
        ret = x
    return ret


def get_initial_layers(input, is_training=True, name='filler'):
    """The initial layer of Enet."""
    conv_init = tf.layers.conv2d(
        inputs=input,
        filters=13,
        kernel_size=[3, 3],
        strides=[2, 2],
        padding='SAME',
        name=name+'init')
    batched = tf.layers.batch_normalization(conv_init, training=is_training)
    activated = tf.nn.relu(batched, name=None)

    max_pooled = tf.nn.max_pool(
        value=input,
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding='SAME',
        name='max_pool_init')

    output = tf.concat(values=[max_pooled, activated], axis=3)
    return output


def get_batch_prelu(input, is_training=True):
    batched = tf.layers.batch_normalization(input, training=is_training)
    output = tf.nn.relu(batched)
    return output


def get_main_layers(input, depth, downsample=False, mode='normal', rate=False, filter_output=False, size_output=False, keep_prob = 0.8, is_training=True, name="filler"):
    """Main layer function"""
    kernel_size = [1, 1]
    stride = [1, 1]
    pooling_indices = 0

    if downsample:
        if mode != 'un':
            kernel_size = [2, 2]
            stride = [2, 2]
            main_track = tf.layers.max_pooling2d(
                inputs=input,
                pool_size=[2, 2],
                strides=[2, 2],
                padding='SAME')

            inputs_shape = input.get_shape().as_list()
            depth_to_pad = abs(inputs_shape[3] - depth)
            paddings = tf.convert_to_tensor([[0, 0], [0, 0], [0, 0], [0, depth_to_pad]])
            main_track = tf.pad(main_track, paddings=paddings)
        elif mode == 'un':
            # main_track = unpool(input, pooling_indices_unpool, output_shape=input_shape)
            input = tf.layers.conv2d(
                inputs=input,
                filters=filter_output,
                kernel_size=kernel_size,
                strides=stride,
                padding='SAME',
                use_bias=False,
                trainable=True,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name=name+'unp')
            input = tf.layers.batch_normalization(input, training=is_training)
            main_track = tf.image.resize_images(input, [size_output, size_output], method=ResizeMethod.BILINEAR)
            print(main_track)

    else:
        main_track = input

    first_layer = tf.layers.conv2d(
        inputs=input,
        filters=16,
        kernel_size=kernel_size,
        strides=stride,
        padding='SAME',
        use_bias=False,
        kernel_initializer=tf.contrib.layers.xavier_initializer(),
        name=name+'1')

    kernel_size = [1, 1]
    stride = [1, 1]

    first_layer = get_batch_prelu(first_layer, is_training)

    if mode == 'normal':
        second_layer = tf.layers.conv2d(
            inputs=first_layer,
            filters=16,
            kernel_size=[3, 3],
            strides=stride,
            padding='SAME',
            use_bias=True,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            name=name+'2')

    elif mode == 'dilated':
        second_layer = tf.layers.conv2d(
            inputs=first_layer,
            filters=16,
            kernel_size=[3, 3],
            padding='SAME',
            dilation_rate=(rate, rate),
            use_bias=True,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            name=name+'2')
    elif mode == 'assymetric':
        second_layer = tf.layers.conv2d(
            inputs=first_layer,
            filters=16,
            kernel_size=[5, 1],
            padding='SAME',
            use_bias=True,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            name=name+'20')
        second_layer = tf.layers.conv2d(
            inputs=second_layer,
            filters=16,
            kernel_size=[1, 5],
            padding='SAME',
            use_bias=True,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            name=name+'21')
    elif mode == 'un':
            net_unpool_shape = main_track.get_shape().as_list()
            if net_unpool_shape[0] == None:
                net_unpool_shape[0] = 1
            output_shape = [net_unpool_shape[0], net_unpool_shape[1], net_unpool_shape[2], 16]
            output_shape = tf.convert_to_tensor(output_shape)
            print(output_shape.get_shape().as_list())
            filter_size = [3, 3, 16, 16]
            filters = tf.get_variable(name='filters_' + str(size_output), shape=filter_size, dtype=tf.float32)

            second_layer = tf.nn.conv2d_transpose(
                first_layer,
                filter=filters,
                strides=[1, 2, 2, 1],
                output_shape=output_shape,
                name=name+'transp')

    second_layer = get_batch_prelu(second_layer, is_training)

    third_layer = tf.layers.conv2d(
        inputs=second_layer,
        filters=depth,
        kernel_size=kernel_size,
        strides=stride,
        padding='SAME',
        use_bias=False,
        kernel_initializer=tf.contrib.layers.xavier_initializer(),
        name=name+'3')

    bi_track = spatial_dropout(third_layer, keep_prob=keep_prob, is_training=is_training)  # keep_prob changes with bottleneck
    output = tf.add(main_track, bi_track)

    output = get_batch_prelu(output, is_training)
    print(output)

    return output, pooling_indices


def Enet(input_image, is_training=True):
    """Main function."""
    if is_training:
        reuse = False
    else:
        reuse = True
    with tf.variable_scope("Enet", reuse=reuse):
        tf.Graph().as_default()
        output = get_initial_layers(input_image, name='init')
        for layer_nbr in range(1, 30):
            if layer_nbr == 1:
                """Bottleneck1.0"""
                depth = 64
                print('Bottleneck1.0')
                print('downsample')
                output, _ = get_main_layers(output, depth=depth, downsample=True, keep_prob=0.8, is_training=is_training, name=str(layer_nbr))
            elif layer_nbr >= 2 and layer_nbr <= 5:
                """Bottleneck1.1-4"""
                output, _ = get_main_layers(output, depth=depth, keep_prob=0.8, is_training=is_training, name=str(layer_nbr))
                print('Bottleneck1.1-4')
                print("normal: ", layer_nbr)
            elif layer_nbr == 6:
                """Bottleneck2.0"""
                depth = 128
                output, _ = get_main_layers(output, depth=depth, downsample=True, is_training=is_training, name=str(layer_nbr))
                print('Bottleneck2.0')
                print('downsample: ', layer_nbr)

            elif layer_nbr == 7 or layer_nbr == 14:
                """Bottleneck2.1"""
                output, _ = get_main_layers(output, depth=depth, is_training=is_training, name=str(layer_nbr))
                print('Bottleneck2.1')
                print('normal stuff: ', layer_nbr)

            elif layer_nbr == 8 or layer_nbr == 15:
                """Bottleneck2.2"""
                output, _ = get_main_layers(output, depth=depth, mode='dilated', rate=2, is_training=is_training, name=str(layer_nbr))
                print('Bottleneck2.2')
                print('dilated_2: ', layer_nbr)

            elif layer_nbr == 9 or layer_nbr == 16:
                """Bottleneck2.3"""
                output, _ = get_main_layers(output, depth=depth, mode='assymetric', is_training=is_training, name=str(layer_nbr))
                print('Bottleneck2.3')
                print('assymetric_5: ', layer_nbr)

            elif layer_nbr == 9 or layer_nbr == 17:
                """Bottleneck2.4"""
                output, _ = get_main_layers(output, depth=depth, mode='dilated', rate=4, is_training=is_training, name=str(layer_nbr))
                print('Bottleneck2.4')
                print('dilated_4: ', layer_nbr)
            elif layer_nbr == 10 or layer_nbr == 18:
                """Bottleneck2.5"""
                output, _ = get_main_layers(output, depth=depth, is_training=is_training, name=str(layer_nbr))
                print('Bottleneck2.5')
                print('normal: ', layer_nbr)
            elif layer_nbr == 11 or layer_nbr == 19:
                """Bottleneck2.6"""
                output, _ = get_main_layers(output, depth=depth, mode='dilated', rate=8, is_training=is_training, name=str(layer_nbr))
                print('Bottleneck2.6')
                print('dilated_8: ', layer_nbr)
            elif layer_nbr == 12 or layer_nbr == 20:
                """Bottleneck2.7"""
                output, _ = get_main_layers(output, depth=depth, mode='assymetric', is_training=is_training, name=str(layer_nbr))
                print('Bottleneck2.7')
                print('assymetric_5: ', layer_nbr)
            elif layer_nbr == 13 or layer_nbr == 21:
                """Bottleneck2.8"""
                output, _ = get_main_layers(output, depth=depth, mode='dilated', rate=16, is_training=is_training, name=str(layer_nbr))
                print('Bottleneck2.8')
                print('dilated_16: ', layer_nbr)
            elif layer_nbr == 22:
                """Bottleneck4.0"""
                depth = 64
                output, _ = get_main_layers(output, depth=depth, downsample=True, mode='un', filter_output=depth, size_output=int(np.round(75/4)), is_training=is_training, name=str(layer_nbr))
                print('Bottleneck4.0')
                print('downsampling_unpool: ', layer_nbr)
            elif layer_nbr == 23 or layer_nbr == 24:
                """Bottleneck4.1-2"""
                depth = 64
                output, _ = get_main_layers(output, depth=depth, is_training=is_training, name=str(layer_nbr))
                print('Bottleneck4.1-2')
                print('decodder: ', layer_nbr)
                print("gg")
            elif layer_nbr == 25:
                """Bottleneck5.0"""
                depth = 16
                output, _ = get_main_layers(output, depth=depth, downsample=True, mode='un', filter_output=depth, size_output=int(np.round(75/2)), is_training=is_training, name=str(layer_nbr))
                print('Bottleneck5.0')
                print('downsampling_unpool: ', layer_nbr)
            elif layer_nbr == 26:
                """Bottleneck5.1"""
                output, _ = get_main_layers(output, depth=depth, is_training=is_training, name=str(layer_nbr))
                print('Bottleneck5.1')
                print('decodder: ', layer_nbr)
            elif layer_nbr == 27:
                """Bottleneck5.2"""
                depth = 16
                output, _ = get_main_layers(output, depth=depth, downsample=True, mode='un', filter_output=depth, size_output=75, is_training=is_training, name=str(layer_nbr))
                print('Bottleneck5.2')
                print('downsampling_unpool: ', layer_nbr)
            elif layer_nbr == 28:

                if True:
                    output = tf.contrib.layers.flatten(output)
                    if True:
                        if is_training:
                            output = tf.nn.dropout(
                                output,
                                keep_prob=0.8)
                        output = tf.layers.dense(
                            inputs=output,
                            units=1028,
                            name='out1')
                        output = get_batch_prelu(output, is_training)
                        if is_training:
                            output = tf.nn.dropout(
                                output,
                                keep_prob=0.8)
                        if is_training:
                            output = tf.nn.dropout(
                                output,
                                keep_prob=0.8)
                        output = tf.layers.dense(
                            inputs=output,
                            units=256,
                            name='out3')
                        output = get_batch_prelu(output, is_training)
                        if is_training:
                            output = tf.nn.dropout(
                                output,
                                keep_prob=0.8)
                        output = tf.layers.dense(
                            inputs=output,
                            units=128,
                            name='out4')
                        output = tf.layers.batch_normalization(output, training=is_training)
                        output = tf.nn.sigmoid(output)
                        #output = tf.nn.softmax(output)
                        print(layer_nbr)
                        print(output)



                    output = tf.layers.dense(
                        inputs=output,
                        units=1,
                        name='out6')
                    #output = tf.layers.batch_normalization(output, training=is_training)
                    probabilities = tf.nn.sigmoid(output)
                    #probabilities = tf.nn.softmax(output)
                    print(layer_nbr + 1)
                    print(output)

        return output, probabilities


def main(x_input, y_input, batch_size = 2):
    """Main function."""
    input_image = tf.placeholder(shape=[batch_size, 75, 75, 3], dtype=tf.float32)
    ground_truth = tf.placeholder(shape=[batch_size, 1], dtype=tf.float32)
    prediction = Enet(input_image, is_training=True)
    loss = tf.losses.mean_squared_error(ground_truth, prediction)
    #loss_collection = tf.losses.add_loss(loss)
    #losses = tf.losses.get_total_loss()
    optimizer = tf.train.AdamOptimizer(0.1)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(loss, global_step=global_step)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #sess.run(tf.local_variables_initializer())

        for step in range(0, 100, 2):
            _, loss_train, pred = sess.run([train_op, loss, prediction], feed_dict={input_image: x_input[step:step+batch_size], ground_truth: y_input[step:step+batch_size]})
            print("step: ", step)
            print("loss: ", loss_train)
            print("pred: ", pred)

if __name__ == '__main__':
    size = 100
    image = np.ones(([size, 75, 75, 3]))
    for i in range(size):
        image = np.ones(([i, 75, 75, 3]))*i
    gt = np.ones(([size, 1]))*0.3
    main(np.array(image, dtype=np.float32), np.array(gt, dtype=np.float32))
