'''
MSE_ResNet8 (Modified Squeeze and Excite Residual Network)
Creator: Subhrasankar Chatterjee

'''
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import cbof
import numpy as np
from keras.models import Model
from keras.layers import (
    Input,
    Activation,
    merge,
    Dense,
    Flatten,
    Dropout
)
from keras.layers.convolutional import (
    Convolution3D,
    MaxPooling3D,
    AveragePooling3D,
    Conv3D
)

from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import MaxPooling2D
from keras.layers import GlobalAveragePooling3D
from keras.layers import GlobalMaxPooling2D
from keras.layers import GlobalMaxPooling3D

from keras.layers import Conv2D
from keras.layers import add
from keras.layers import multiply
from keras.regularizers import l2
from keras.utils import conv_utils
from keras.utils.data_utils import get_file
from keras.engine.topology import get_source_inputs
from keras_applications.imagenet_utils import _obtain_input_shape
from keras_applications.resnet50 import preprocess_input
from keras_applications.imagenet_utils import decode_predictions
from keras.regularizers import l2
from keras import regularizers
from keras import backend as K

from contextlib import redirect_stdout



def _handle_dim_ordering():
    global CONV_DIM1
    global CONV_DIM2
    global CONV_DIM3
    global CHANNEL_AXIS
    if K.image_dim_ordering() == 'tf':
        CONV_DIM1 = 1
        CONV_DIM2 = 2
        CONV_DIM3 = 3
        CHANNEL_AXIS = 4
    else:
        CHANNEL_AXIS = 1
        CONV_DIM1 = 2
        CONV_DIM2 = 3
        CONV_DIM3 = 4

def _bn_relu(input):
	norm = BatchNormalization(axis=CHANNEL_AXIS)(input)
	return Activation("relu")(norm)

def _conv_bn_relu(**params):
    nb_filter = params["nb_filter"]
    kernel_dim1 = params["kernel_dim1"]
    kernel_dim2 = params["kernel_dim2"]
    kernel_dim3 = params["kernel_dim3"]
    subsample = params.setdefault("subsample", (1, 1, 1))
    init = params.setdefault("init", "he_normal")
    border_mode = params.setdefault("border_mode", "same")
    W_regularizer = params.setdefault("W_regularizer", regularizers.l2(1.e-4))
    def f(input):
        conv = Conv3D(kernel_initializer=init,strides=subsample,kernel_regularizer= W_regularizer, filters=nb_filter, kernel_size=(kernel_dim1,kernel_dim2,kernel_dim3))(input)
        return _bn_relu(conv)

    return f

def _bn_relu_conv(**params):
    nb_filter = params["nb_filter"]
    kernel_dim1 = params["kernel_dim1"]
    kernel_dim2 = params["kernel_dim2"]
    kernel_dim3 = params["kernel_dim3"]
    subsample = params.setdefault("subsample", (1,1,1))
    init = params.setdefault("init", "he_normal")
    border_mode = params.setdefault("border_mode", "same")
    W_regularizer = params.setdefault("W_regularizer", l2(1.e-4))

    def f(input):
        activation = _bn_relu(input)
        return Conv3D(kernel_initializer=init, strides=subsample, kernel_regularizer=W_regularizer,
                          filters=nb_filter, kernel_size=(kernel_dim1, kernel_dim2, kernel_dim3), padding=border_mode)(activation)

    return f

def _shortcut(input, residual):
    stride_dim1 = (input._keras_shape[CONV_DIM1]+1) // residual._keras_shape[CONV_DIM1]
    stride_dim2 = (input._keras_shape[CONV_DIM2]+1) // residual._keras_shape[CONV_DIM2]
    stride_dim3 = (input._keras_shape[CONV_DIM3]+1) // residual._keras_shape[CONV_DIM3]
    equal_channels = residual._keras_shape[CHANNEL_AXIS] == input._keras_shape[CHANNEL_AXIS]

    shortcut = input
    print("input shape:", input._keras_shape)

    shortcut = Conv3D(kernel_initializer="he_normal", strides=(stride_dim1, stride_dim2, stride_dim3), kernel_regularizer=regularizers.l2(0.0001),filters=residual._keras_shape[CHANNEL_AXIS], kernel_size=(1, 1, 1), padding='valid')(input)
    shortcut = squeeze_excite_block(shortcut)
    return add([shortcut, residual])
	
def _shortcut_spc(input, residual):
    stride_dim1 = (input._keras_shape[CONV_DIM1]+1) // residual._keras_shape[CONV_DIM1]
    stride_dim2 = (input._keras_shape[CONV_DIM2]+1) // residual._keras_shape[CONV_DIM2]
    stride_dim3 = (input._keras_shape[CONV_DIM3]+1) // residual._keras_shape[CONV_DIM3]
    equal_channels = residual._keras_shape[CHANNEL_AXIS] == input._keras_shape[CHANNEL_AXIS]

    shortcut = input
    print("input shape:", input._keras_shape)
    shortcut = squeeze_excite_block(residual)
    return add([shortcut, residual])
	
	
def basic_block(nb_filter, init_subsample=(1, 1, 1), is_first_block_of_first_layer=False):
    def f(input):

        if is_first_block_of_first_layer:
            conv1 = Conv3D(kernel_initializer="he_normal", strides=init_subsample, kernel_regularizer=regularizers.l2(0.0001),
                          filters=nb_filter, kernel_size=(3, 3, 1), padding='same')(input)
        else:
            conv1 = _bn_relu_conv(nb_filter=nb_filter, kernel_dim1=3, kernel_dim2=3, kernel_dim3=1, subsample=init_subsample)(input)

        residual = _bn_relu_conv(nb_filter=nb_filter, kernel_dim1=3, kernel_dim2=3, kernel_dim3=1)(conv1)
        return _shortcut(input, residual)

    return f

def basic_block_spc(nb_filter, init_subsample=(1, 1, 1), is_first_block_of_first_layer=False):
    def f(input):

        if is_first_block_of_first_layer:
            conv1 = Conv3D(kernel_initializer="he_normal", strides=init_subsample, kernel_regularizer=regularizers.l2(0.0001),
                          filters=nb_filter, kernel_size=(3, 3, 1), padding='same')(input)
        else:
            conv1 = _bn_relu_conv(nb_filter=nb_filter, kernel_dim1=3, kernel_dim2=3, kernel_dim3=1, subsample=init_subsample)(input)

        residual = _bn_relu_conv(nb_filter=nb_filter, kernel_dim1=3, kernel_dim2=3, kernel_dim3=1)(conv1)
        return _shortcut_spc(input, residual)

    return f

def squeeze_excite_block(input, ratio=16):
    init = input
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = init._keras_shape[channel_axis]
    se_shape = (1, 1, filters)

    #se = GlobalAveragePooling3D()(init)
    se = GlobalMaxPooling3D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    x = multiply([init, se])
    return x
	
def SE_ResNet8(input_shape, num_outputs, codebooks):
	_handle_dim_ordering()
	if len(input_shape) != 4:
		raise Exception("Input shape should be a tuple (nb_channels, kernel_dim1, kernel_dim2, kernel_dim3)")
	
	if K.image_dim_ordering() == 'tf':
		input_shape = (input_shape[1], input_shape[2],input_shape[3], input_shape[0])
	input = Input(shape = input_shape)
	
	conv1 = _conv_bn_relu(nb_filter=64, kernel_dim1=1, kernel_dim2=1, kernel_dim3=7, subsample=(1, 1, 2))(input)
	
	conv2 = basic_block_spc(64, is_first_block_of_first_layer = True)(conv1)
	bn3 = _bn_relu(conv2)
	#bn3 = _bn_relu(bn3_1)
	bn4 = _conv_bn_relu(nb_filter=64, kernel_dim1=1, kernel_dim2=1, kernel_dim3=bn3._keras_shape[CONV_DIM3], subsample=(1, 1, 2) , border_mode='valid')(bn3)
	resh = Reshape((bn4._keras_shape[CONV_DIM1],bn4._keras_shape[CONV_DIM2],bn4._keras_shape[CHANNEL_AXIS],1))(bn4)
	conv4 = _conv_bn_relu(nb_filter=64, kernel_dim1=3, kernel_dim2=3, kernel_dim3=64,
                              subsample=(1, 1, 1))(resh)
	conv5 = basic_block(64, is_first_block_of_first_layer = True)(conv4)
	bn5 = _bn_relu(conv5)
	bn6 = _bn_relu(bn5)

	#pool2 = AveragePooling3D(pool_size=(bn6._keras_shape[CONV_DIM1],
                                            #bn6._keras_shape[CONV_DIM2],
                                            #bn6._keras_shape[CONV_DIM3],),strides=(1, 1, 1))(bn6)
	
	#flatten1 = Flatten()(bn6)
	print(bn6.shape)
	bn6 = Reshape((bn6.shape[1],bn6.shape[2],bn6.shape[4]))(bn6)
	print(bn6.shape)
	
	#bn6 = K.squeeze(bn6,axis = 3)
	flatten1 = cbof.BoF_Pooling(codebooks, spatial_level=0)(bn6)
	drop1 = Dropout(0.5)(flatten1)
	dense = Dense(units=num_outputs, activation="softmax", kernel_initializer="he_normal")(drop1)	
	model = Model( inputs = input , outputs = dense)
	model.summary()	
	return model
SE_ResNet8((1,15,15,30),16,64)
