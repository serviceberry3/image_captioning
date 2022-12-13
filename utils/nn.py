import tensorflow.compat.v1 as tf

#import tensorflow.contrib.layers as layers

#Keras is a high-level, deep learning API developed by Google for implementing neural networks
import tensorflow.keras as keras

class NN(object):
    def __init__(self, config):
        self.config = config
        self.is_train = True if config.phase == 'train' else False
        self.train_cnn = self.is_train and config.train_cnn
        self.prepare()

    def prepare(self):
        """ Setup the weight initalizers and regularizers. Regularizers allow you to apply penalties on layer parameters or 
        layer activity during optimization. These penalties are summed into the loss function that the network optimizes. """
        config = self.config

        #CHANGE: TF1 to TF2
        #self.conv_kernel_initializer = layers.xavier_initializer()
        self.conv_kernel_initializer = keras.initializers.glorot_normal


        if self.train_cnn and config.conv_kernel_regularizer_scale > 0:
            #CHANGE: TF1 to TF2
            #self.conv_kernel_regularizer = layers.l2_regularizer(scale = config.conv_kernel_regularizer_scale)
            self.conv_kernel_regularizer = keras.regularizers.l2(l2 = config.conv_kernel_regularizer_scale)
        else:
            self.conv_kernel_regularizer = None

        if self.train_cnn and config.conv_activity_regularizer_scale > 0:
            #CHANGE: TF1 to TF2
            #self.conv_activity_regularizer = layers.l1_regularizer(scale = config.conv_activity_regularizer_scale)
            self.conv_activity_regularizer = keras.regularizers.l1(l1 = config.conv_kernel_regularizer_scale)
        else:
            self.conv_activity_regularizer = None

        self.fc_kernel_initializer = tf.random_uniform_initializer(
            minval = -config.fc_kernel_initializer_scale,
            maxval = config.fc_kernel_initializer_scale)

        if self.is_train and config.fc_kernel_regularizer_scale > 0:
            #CHANGE: TF1 to TF2
            #self.fc_kernel_regularizer = layers.l2_regularizer(scale = config.fc_kernel_regularizer_scale)
            self.fc_kernel_regularizer = keras.regularizers.l2(l2 = config.fc_kernel_regularizer_scale)
        else:
            self.fc_kernel_regularizer = None

        if self.is_train and config.fc_activity_regularizer_scale > 0:
            #CHANGE: TF1 to TF2
            #self.fc_activity_regularizer = layers.l1_regularizer(scale = config.fc_activity_regularizer_scale)
            self.fc_activity_regularizer = keras.regularizers.l1(l1 = config.fc_activity_regularizer_scale)
        else:
            self.fc_activity_regularizer = None

    def conv2d(self,
               inputs,
               filters,
               kernel_size = (3, 3),
               strides = (1, 1),
               activation = tf.nn.relu,
               use_bias = True,
               name = None):

        """ 2D Convolution layer. """
        if activation is not None:
            activity_regularizer = self.conv_activity_regularizer
        else:
            activity_regularizer = None

        #CHANGE: TF1 to TF2
        return tf.layers.conv2d(
            inputs = inputs,
            filters = filters,
            kernel_size = kernel_size,
            strides = strides,
            padding='same',
            activation = activation,
            use_bias = use_bias,
            trainable = self.train_cnn,
            kernel_initializer = self.conv_kernel_initializer,
            kernel_regularizer = self.conv_kernel_regularizer,
            activity_regularizer = activity_regularizer,
            name = name)


    def max_pool2d(self,
                   inputs,
                   pool_size = (2, 2),
                   strides = (2, 2),
                   name = None):

        """ 2D Max Pooling layer. """
        #CHANGE: TF1 to TF2
        return tf.layers.max_pooling2d(
            inputs = inputs,
            pool_size = pool_size,
            strides = strides,
            padding='same',
            name = name)

    #dense layer convenience fxn
    #just your regular densely-connected NN layer
    def dense(self,
              inputs,
              units,
              activation = tf.tanh,
              use_bias = True,
              name = None):

        """ Fully-connected layer. """
        if activation is not None:
            activity_regularizer = self.fc_activity_regularizer
        else:
            activity_regularizer = None

        #CHANGE: TF1 to TF2
        return tf.layers.dense(
            inputs = inputs,
            units = units,
            activation = activation,
            use_bias = use_bias,
            trainable = self.is_train,
            kernel_initializer = self.fc_kernel_initializer,
            kernel_regularizer = self.fc_kernel_regularizer,
            activity_regularizer = activity_regularizer,
            name = name)


    #dropout layer convenience fxn
    #a dropout layer applies Dropout to the input
    def dropout(self,
                inputs,
                name = None):
        """ Dropout layer. """
        #CHANGE: TF1 to TF2
        return tf.layers.dropout(
            inputs = inputs,
            rate = self.config.fc_drop_rate,
            training = self.is_train)


    #batchnorm layer convenience fxn
    #a batchnorm layer is a layer that normalizes its inputs
    def batch_norm(self,
                   inputs,
                   name = None):

        """ Batch normalization layer. """
        return tf.layers.batch_normalization(
            inputs = inputs,
            training = self.train_cnn,
            trainable = self.train_cnn,
            name = name
        )

    
    #reshape layer convenience fxn
    def reshape(self,
                inputs,
                new_shape,
                name = None):
        return keras.layers.Reshape(target_shape = new_shape)
