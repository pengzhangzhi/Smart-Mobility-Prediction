import tensorflow as tf
from keras.models import Model
from keras.layers import (
    Input,
    Activation,
    Flatten,
    Concatenate,
    Dense,
    Reshape,
    Conv2D,
    BatchNormalization,
    TimeDistributed,
    Conv2DTranspose,
    LSTM,
    Add
)
from keras.optimizers import Adam
import numpy as np

import src.metrics as metrics

class MultiplicativeUnit():
    """Initialize the multiplicative unit.
    Args:
       layer_name: layer names for different multiplicative units.
       filter_size: int tuple of the height and width of the filter.
       num_hidden: number of units in output tensor.
    """
    def __init__(self, layer_name, num_hidden, filter_size):
        self.layer_name = layer_name
        self.num_features = num_hidden
        self.filter_size = filter_size

    def __call__(self, h, reuse=False):
        with tf.compat.v1.variable_scope(self.layer_name, reuse=reuse):
            g1 = Conv2D(
                self.num_features, self.filter_size, padding='same', activation=tf.sigmoid,
                # kernel_initializer=tf.contrib.layers.xavier_initializer(),
                )(h)
            g2 = Conv2D(
                self.num_features, self.filter_size, padding='same', activation=tf.sigmoid,
                # kernel_initializer=tf.contrib.layers.xavier_initializer(),
                )(h)
            g3 = Conv2D(
                self.num_features, self.filter_size, padding='same', activation=tf.sigmoid,
                # kernel_initializer=tf.contrib.layers.xavier_initializer(),
                )(h)
            u = Conv2D(
                self.num_features, self.filter_size, padding='same', activation=tf.tanh,
                # kernel_initializer=tf.contrib.layers.xavier_initializer(),
                )(h)
            g2_h = tf.multiply(g2, h)
            g3_u = tf.multiply(g3, u)
            mu = tf.multiply(g1, tf.tanh(g2_h + g3_u))
            return mu

class CMU():
    """Initialize the causal multiplicative unit.
    Args:
       layer_name: layer names for different causal multiplicative unit.
       filter_size: int tuple of the height and width of the filter.
       num_hidden: number of units in output tensor.
    """
    def __init__(self, layer_name, num_hidden, filter_size):
        self.layer_name = layer_name
        self.num_features = num_hidden
        self.filter_size = filter_size

    def __call__(self, h1, h2, stride=False, reuse=False):
        with tf.compat.v1.variable_scope(self.layer_name, reuse=reuse):
            hl = MultiplicativeUnit('multiplicative_unit_1', self.num_features, self.filter_size)(h1, reuse=reuse)
            if not stride:
                hl = MultiplicativeUnit('multiplicative_unit_1', self.num_features, self.filter_size)(hl, reuse=True)
            hr = MultiplicativeUnit('multiplicative_unit_2', self.num_features, self.filter_size)(h2, reuse=reuse)
            h = tf.add(hl, hr)
            h_sig = Conv2D(
                self.num_features, self.filter_size, padding='same', activation=tf.sigmoid,
                # kernel_initializer=tf.contrib.layers.xavier_initializer(),
                )(h)
            h_tan = Conv2D(
                self.num_features, self.filter_size, padding='same', activation=tf.tanh,
                # kernel_initializer=tf.contrib.layers.xavier_initializer(),
                )(h)
            h = tf.multiply(h_sig, h_tan)
            return h

def predcnn_perframe(xs, num_hidden, filter_size, input_length, reuse):
    with tf.compat.v1.variable_scope('frame_prediction', reuse=reuse):
        for i in range(input_length-1):
            temp = []
            for j in range(input_length-i-1):
                h1 = xs[j]
                h2 = xs[j+1]
                h = CMU('causal_multiplicative_unit_'+str(i+1), num_hidden, filter_size)(h1, h2, stride=False, reuse=bool(temp))
                temp.append(h)
            xs = temp
        return xs[0]

def my_conv(filters, activation, kernel_size=3, time_distributed=False):
    def f(input_layer):
        if (time_distributed):
            l = TimeDistributed(Conv2D(filters, kernel_size, padding='same', activation=activation))(input_layer)
            l = TimeDistributed(BatchNormalization())(l)
            return l
        else:
            l = Conv2D(filters, kernel_size, padding='same', activation=activation)(input_layer)
            l = BatchNormalization()(l)
            return l
    return f

def my_downsampling(input_layer, filters):
    l = TimeDistributed(Conv2D(filters, (2,2), (2,2), activation='relu'))(input_layer)
    l = TimeDistributed(BatchNormalization())(l)
    return l

def my_conv_transpose(input_layer, skip_connection_layer):
    l = Conv2DTranspose(input_layer.shape[-1], (2,2), (2,2))(input_layer)
    skl = skip_connection_layer[:,-1,:,:,:] # extract features from last image
    l = Add()([l, skl])
    l = Activation('relu')(l)
    l = BatchNormalization()(l)
    return l

def _residual_unit(filters, num_res, kernel_size, td):
    def f(input_layer):
        residual = my_conv(filters, 'relu', kernel_size, time_distributed=td)(input_layer)
        for _ in range(num_res-1):
            residual = my_conv(filters, 'relu', kernel_size, time_distributed=td)(residual)
        
        return Add()([input_layer, residual])
    return f

def TSresUnits2D(filters, num_res, kernel_size, time_distributed, repetations=1):
    def f(input_layer):
        for i in range(repetations):
            input_layer = _residual_unit(filters, num_res, kernel_size, time_distributed)(input_layer)
        return input_layer
    return f

def my_model(len_c, len_p, len_t, nb_flow=2, map_height=32, map_width=32,
             external_dim=8, encoder_blocks=2, filters=[64,64,64,16], kernel_size=3,
             num_res=2):
    # len(filters) has to be encoder blocks + 2 (first and last conv)

    main_inputs = []
    #ENCODER
    # input layer tx32x32x2
    t = len_c+len_p*2+len_t*2
    input = Input(shape=((t, map_height, map_width, nb_flow)))
    main_inputs.append(input)
    x = input

    # build encoder blocks
    # first conv
    x = my_conv(filters[0], 'relu', kernel_size, time_distributed=True)(x)
    # residual blocks
    skip_connection_layers = []
    for i in range(1, encoder_blocks+1): # es: encoder_blocks=2 -> i from 1 to 2       
        # res unit
        x = TSresUnits2D(filters[i], num_res, kernel_size, time_distributed=True)(x)
        # append layer to skip connection list
        skip_connection_layers.append(x)
        # downsampling
        x = my_downsampling(x, x.shape[-1])

    # last convolution tx4x4x16
    x = my_conv(filters[-1], 'relu', kernel_size, time_distributed=True)(x)
    s = x.shape
    # print(s)

    # CMUs
    list_features = [x[:,i,:,:,:] for i in range(x.shape[1])]
    x = predcnn_perframe(list_features, s[-1], kernel_size, t, reuse=False)
    x = Reshape((s[2:]))(x)

    # merge external features
    if external_dim != None and external_dim > 0:
        # external input
        external_input = Input(shape=(external_dim,))
        main_inputs.append(external_input)
        embedding = Dense(units=10, activation='relu')(external_input)
        h1 = Dense(units=s[2]*s[3]*s[4], activation='relu')(embedding)
        external_output = Reshape((s[2], s[3], s[4]))(h1)
        x = Add()([x, external_output])

    # build decoder blocks
    # first conv decoder
    x = my_conv(filters[-2], 'relu', kernel_size)(x)
    # decoder
    for i in reversed(range(1, encoder_blocks+1)):
        # conv_transpose + skip_conn + relu + bn
        x = my_conv_transpose(x, skip_connection_layers[i-1])
        # conv + relu + bn
        x = TSresUnits2D(filters[i], num_res, kernel_size, time_distributed=False)(x)
    #final conv
    output = my_conv(nb_flow, 'tanh')(x)


    return Model(main_inputs, output)

def build_model(len_c, len_p, len_t, nb_flow=2, map_height=32, map_width=32,
                external_dim=8, encoder_blocks=2, filters=[32,64,64,16],
                kernel_size=3, num_res=2, lr=0.0001, save_model_pic=None):
    model = my_model(len_c, len_p, len_t, nb_flow, map_height, map_width,
                     external_dim, encoder_blocks, filters, kernel_size, num_res)
    adam = Adam(lr=lr)
    model.compile(loss='mse', optimizer=adam, metrics=[metrics.rmse])
    # model.summary()
    if (save_model_pic):
        from keras.utils.vis_utils import plot_model
        plot_model(model, to_file=f'{save_model_pic}.png', show_shapes=True)
    return model