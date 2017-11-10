import tensorflow as tf
from tensorflow.contrib import layers
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.merge import add
from keras import regularizers

#########################################
# ADD HERE YOUR NETWORK                 #
# BUILD IT WITH PURE TENSORFLOW OR KERAS#
#########################################

# NOTE: IF USING KERAS, YOU MIGHT HAVE PROBLEMS WITH BATCH-NORMALIZATION

def resnet8(img_input, output_dim, scope='Prediction', reuse=False, log=False):
    """
    Define model architecture in Keras.

    # Arguments
       img_input: Batch of input images
       output_dim: Number of output dimensions (cardinality of classification)
       scope: Variable scope in which all variables will be saved
       reuse: Whether to reuse already initialized variables

    # Returns
       logits: Logits on output trajectories
    """

    img_input = Input(tensor=img_input)
    with tf.variable_scope(scope, reuse=reuse):
        x1 = Conv2D(32, (5, 5), strides=[2,2], padding='same')(img_input)
        x1 = MaxPooling2D(pool_size=(3, 3), strides=[2,2])(x1)

        # First residual block
        x2 = Activation('relu')(x1)
        x2 = Conv2D(32, (3, 3), strides=[2,2], padding='same',
                    kernel_initializer="he_normal",
                    kernel_regularizer=regularizers.l2(1e-4))(x2)

        x2 = Activation('relu')(x2)
        x2 = Conv2D(32, (3, 3), padding='same',
                    kernel_initializer="he_normal",
                    kernel_regularizer=regularizers.l2(1e-4))(x2)

        x1 = Conv2D(32, (1, 1), strides=[2,2], padding='same')(x1)
        x3 = add([x1, x2])

        # Second residual block
        x4 = Activation('relu')(x3)
        x4 = Conv2D(64, (3, 3), strides=[2,2], padding='same',
                    kernel_initializer="he_normal",
                    kernel_regularizer=regularizers.l2(1e-4))(x4)

        x4 = Activation('relu')(x4)
        x4 = Conv2D(64, (3, 3), padding='same',
                    kernel_initializer="he_normal",
                    kernel_regularizer=regularizers.l2(1e-4))(x4)

        x3 = Conv2D(64, (1, 1), strides=[2,2], padding='same')(x3)
        x5 = add([x3, x4])

        # Third residual block
        x6 = Activation('relu')(x5)
        x6 = Conv2D(128, (3, 3), strides=[2,2], padding='same',
                    kernel_initializer="he_normal",
                    kernel_regularizer=regularizers.l2(1e-4))(x6)

        x6 = Activation('relu')(x6)
        x6 = Conv2D(128, (3, 3), padding='same',
                    kernel_initializer="he_normal",
                    kernel_regularizer=regularizers.l2(1e-4))(x6)

        x5 = Conv2D(128, (1, 1), strides=[2,2], padding='same')(x5)
        x7 = add([x5, x6])

        x = Flatten()(x7)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x)

        # Output channel
        logits = Dense(output_dim)(x)

    if log:
        model = Model(inputs=[img_input], outputs=[logits])
        print(model.summary())

    return logits

def atari_model(img_in, output_dim, scope='mynet', reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        out = img_in
        with tf.variable_scope("convnet"):
            # original architecture
            out = layers.convolution2d(out, num_outputs=32, kernel_size=8, stride=4, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=4, stride=2, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu)
        out = layers.flatten(out)
        with tf.variable_scope("action_value"):
            out = layers.fully_connected(out, num_outputs=512,         activation_fn=tf.nn.relu)
            out = layers.fully_connected(out, num_outputs=output_dim, activation_fn=None)

        return out
def resnet8_tf(img_in, output_dim, is_training, l2_reg_scale,
               scope='Prediction', reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        x1 = layers.convolution2d(img_in, num_outputs=32, kernel_size=5,
                                  stride=2, padding='SAME')
        x1 = layers.max_pool2d(x1, kernel_size=3, stride=2)

        # First residual block
        x2 = tf.nn.relu(x1)
        x2 = layers.convolution2d(x2, num_outputs=32, kernel_size=3,
                                  padding='SAME', stride=2,
            weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg_scale),
                                  activation_fn=tf.nn.relu)

        x2 = layers.convolution2d(x2, num_outputs=32, kernel_size=3,
                                  padding='SAME',
            weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg_scale))

        x1 = layers.convolution2d(x1, num_outputs=32, kernel_size=1, stride=2,
                                  padding='SAME')
        x3 = x2 + x1 # Shortcut connection

        # Second residual block
        x4 = tf.nn.relu(x3)
        x4 = layers.convolution2d(x4, num_outputs=64, kernel_size=3,
                                  padding='SAME', stride=2,
            weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg_scale),
                                  activation_fn=tf.nn.relu)

        x4 = layers.convolution2d(x4, num_outputs=64, kernel_size=3,
                                  padding='SAME',
            weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg_scale))

        x3 = layers.convolution2d(x3, num_outputs=64, kernel_size=1, stride=2,
                                  padding='SAME')
        x5 = x4 + x3

        #  Third residual block
        x6 = tf.nn.relu(x5)
        x6 = layers.convolution2d(x6, num_outputs=128, kernel_size=3,
                                  padding='SAME', stride=2,
            weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg_scale),
                                  activation_fn=tf.nn.relu)

        x6 = layers.convolution2d(x6, num_outputs=128, kernel_size=3,
                                  padding='SAME',
            weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg_scale))

        x5 = layers.convolution2d(x5, num_outputs=128, kernel_size=1, stride=2,
                                  padding='SAME')
        x7 = tf.nn.relu(x6 + x5)

        x = layers.flatten(x7)
        x = layers.dropout(x, keep_prob=0.5, is_training=is_training)
        x = layers.fully_connected(x, num_outputs=256, activation_fn=tf.nn.relu,
            weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg_scale))

        # Final Output
        logits = layers.fully_connected(x, num_outputs = output_dim,
                                        activation_fn=None,
            weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg_scale))

    return logits
