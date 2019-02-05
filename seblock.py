import tensorflow as tf

__all__ = [
    'squeeze_and_excitation_block',
]


def squeeze_and_excitation_block(input_X, out_dim, reduction_ratio=16, layer_name='SE-block'):
    """Squeeze-and-Excitation (SE) Block

    SE block to perform feature recalibration - a mechanism that allows
    the network to perform feature recalibration, through which it can
    learn to use global information to selectively emphasise informative
    features and suppress less useful ones
    """


    with tf.name_scope(layer_name):

        # Squeeze: Global Information Embedding
        squeeze = tf.nn.avg_pool(input_X, ksize=[1, *input_X.shape[1:3], 1], strides=[1, 1, 1, 1], padding='valid', name='squeeze')

        # Excitation: Adaptive Feature Recalibration
        ## Dense (Bottleneck) -> ReLU
        excitation = tf.layers.dense(squeeze, units=out_dim/reduction_ratio, name='excitation-bottleneck')
        excitation = tf.nn.relu(excitation, name='excitation-bottleneck-relu')

        ## Dense -> Sigmoid
        excitation = tf.layers.dense(excitation, units=out_dim, name='excitation')
        excitation = tf.nn.sigmoid(excitation, name='excitation-sigmoid')

        # Scaling
        scaler = tf.reshape(excitation, shape=[-1, 1, 1, out_dim], name='scaler')

        return input_X * scaler
