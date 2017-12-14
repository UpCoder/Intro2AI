'''
You are going to implement the CNN model from paper 'End to End Learning for Self-Driving Cars'.
Write the model below.
'''
import tensorflow as tf


def batch_norm(x):
    '''Batch normlization(I didn't include the offset and scale)
    '''
    epsilon = 1e-3
    batch_mean, batch_var = tf.nn.moments(x, [0])
    x = tf.nn.batch_normalization(x,
                                  mean=batch_mean,
                                  variance=batch_var,
                                  offset=None,
                                  scale=None,
                                  variance_epsilon=epsilon)
    return x


def conv(name, input_tensor, out_channel, k_size, stride=[1, 1, 1, 1], padding=True):
    input_channel = input_tensor.get_shape()[-1]
    with tf.variable_scope(name):
        weights = tf.get_variable(
            'weights',
            shape=[
                k_size[0],
                k_size[1],
                input_channel,
                out_channel,
            ],
            initializer=tf.contrib.layers.xavier_initializer()
        )
        bias = tf.get_variable(
            'biases',
            shape=[
                out_channel
            ],
            initializer=tf.constant_initializer(0.0)
        )
        if padding:
            x = tf.nn.conv2d(
                input_tensor,
                weights,
                strides=stride,
                padding='SAME',
                name='conv'
            )
        else:
            x = tf.nn.conv2d(
                input_tensor,
                weights,
                strides=stride,
                padding='VALID',
                name='conv'
            )
        x = tf.nn.bias_add(
            x, bias,
            name='bias-add'
        )
        x = tf.nn.sigmoid(x, name='relu')
        # x = batch_norm(x)
        return x


def fc(layer_name, x, out_nodes):
    '''Wrapper for fully connected layers with RELU activation as default
    Args:
        layer_name: e.g. 'FC1', 'FC2'
        x: input feature map
        out_nodes: number of neurons for current FC layer
    '''
    shape = x.get_shape()
    if len(shape) == 4:
        size = shape[1].value * shape[2].value * shape[3].value
    else:
        size = shape[-1].value

    with tf.variable_scope(layer_name):
        w = tf.get_variable('weights',
                            shape=[size, out_nodes],
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('biases',
                            shape=[out_nodes],
                            initializer=tf.constant_initializer(0.0))
        flat_x = tf.reshape(x, [-1, size])  # flatten into 1D
        x = tf.nn.bias_add(tf.matmul(flat_x, w), b)
        x = tf.nn.sigmoid(x)
        return x


def flatten(tensor):
    '''
    flatten a tensor
    :param tensor: a tensor with shape [batch size, w, h, channel]
    :return: a tensor with shape [batch size, w*h*channel]
    '''
    shape = tensor.get_shape().as_list()
    size = shape[1] * shape[2] * shape[3]
    return tf.reshape(tensor, [-1, size])


def inference(input_tensor):
    print 'input is ', input_tensor
    layer1_output = conv(
        'layer1-conv',
        input_tensor,
        out_channel=24,
        k_size=[5, 5],
        stride=[1, 2, 2, 1],
        padding=False
    )
    print 'layer1 output is ', layer1_output
    layer2_output = conv(
        'layer2-conv',
        layer1_output,
        out_channel=36,
        k_size=[5, 5],
        stride=[1, 2, 2, 1],
        padding=False
    )
    print 'layer2 output is ', layer2_output
    layer3_output = conv(
        'layer3-conv',
        layer2_output,
        out_channel=48,
        k_size=[5, 5],
        stride=[1, 2, 2, 1],
        padding=False
    )
    print 'layer3 output is ', layer3_output
    layer4_output = conv(
        'layer4-conv',
        layer3_output,
        out_channel=64,
        k_size=[3, 3],
        stride=[1, 1, 1, 1],
        padding=False
    )
    print 'layer4 output is ', layer4_output
    layer5_output = conv(
        'layer5-conv',
        layer4_output,
        out_channel=64,
        k_size=[3, 3],
        stride=[1, 1, 1, 1],
        padding=False
    )
    print 'layer5 output is ', layer5_output
    flatten_output = flatten(layer5_output)
    print 'after flatten, the tensor is ', flatten_output
    layer6_output = fc(
        'layer6-fc',
        flatten_output,
        out_nodes=100,
    )

    layer7_output = fc(
        'layer7-fc',
        layer6_output,
        out_nodes=50,
    )

    layer8_output = fc(
        'layer8-fc',
        layer7_output,
        out_nodes=10
    )

    layer9_output = fc(
        'layer9-fc',
        layer8_output,
        out_nodes=1
    )

    return layer9_output

if __name__ == '__main__':
    input_tensor = tf.placeholder(
        dtype=tf.float32,
        shape=[
            100, 66, 200, 3
        ],
        name='input'
    )
    inference_result = inference(input_tensor)
    print inference_result
