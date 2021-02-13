import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, Concatenate
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError
from tensorflow.nn import max_pool_with_argmax

import tensorflow_addons as tfa
from tensorflow_addons.optimizers import AdamW

from custom_layers import MaxPoolingWithArgmax2D, MaxUnpooling2D

def deepcfd(input_height, input_width, input_channels,
            weight_decay, learning_rate):
    # Shared encoder channel
    inputs = Input(shape=(input_height, input_width, input_channels))

    conv1a = Conv2D(8, (5,5), activation='relu', padding='same')(inputs)
    conv1b = Conv2D(8, (5,5), activation='relu', padding='same')(conv1a)
    # pool1 = MaxPooling2D((2,2))(conv1b)
    pool1, idx1 = MaxPoolingWithArgmax2D(pool_size=(2, 2))(conv1b)

    conv2a = Conv2D(16, (5,5), activation='relu', padding='same')(pool1)
    conv2b = Conv2D(16, (5,5), activation='relu', padding='same')(conv2a)
    # pool2 = MaxPooling2D((2,2))(conv2b)
    pool2, idx2 = MaxPoolingWithArgmax2D(pool_size=(2, 2))(conv2b)

    conv3a = Conv2D(32, (5,5), activation='relu', padding='same')(pool2)
    conv3b = Conv2D(32, (5,5), activation='relu', padding='same')(conv3a)
    # pool3 = MaxPooling2D((2,2))(conv3b)
    pool3, idx3 = MaxPoolingWithArgmax2D(pool_size=(2, 2))(conv3b)

    conv4a = Conv2D(32, (5,5), activation='relu', padding='same')(pool3)
    conv4b = Conv2D(32, (5,5), activation='relu', padding='same')(conv4a)
    # pool4 = MaxPooling2D((2,2))(conv4b)
    # pool4, idx4 = MaxPoolingWithArgmax2D(pool_size=(2, 2))(conv4b)
    
    # Separate Ux decoder channel
    # upsamp4_ux = UpSampling2D((2,2))(pool4)
    # unpool4_ux = MaxUnpooling2D(pool_size=(2, 2), out_shape=conv4b.shape)([pool4, idx4])
    concat4_ux = Concatenate()([conv4a, conv4b])
    deconv4a_ux = Conv2DTranspose(32, (5,5), activation='relu', padding='same')(concat4_ux)
    deconv4b_ux = Conv2DTranspose(32, (5,5), activation='relu', padding='same')(deconv4a_ux)

    # upsamp3_ux = UpSampling2D((2,2))(deconv4b_ux)
    unpool3_ux = MaxUnpooling2D(pool_size=(2, 2), out_shape=conv3b.shape)([deconv4b_ux, idx3])
    concat3_ux = Concatenate()([conv3b, unpool3_ux])
    deconv3a_ux = Conv2DTranspose(32, (5,5), activation='relu', padding='same')(concat3_ux)
    deconv3b_ux = Conv2DTranspose(16, (5,5), activation='relu', padding='same')(deconv3a_ux)

    # upsamp2_ux = UpSampling2D((2,2))(deconv3b_ux)
    unpool2_ux = MaxUnpooling2D(pool_size=(2, 2), out_shape=conv2b.shape)([deconv3b_ux, idx2])
    concat2_ux = Concatenate()([conv2b, unpool2_ux])
    deconv2a_ux = Conv2DTranspose(16, (5,5), activation='relu', padding='same')(concat2_ux)
    deconv2b_ux = Conv2DTranspose(8, (5,5), activation='relu', padding='same')(deconv2a_ux)

    # upsamp1_ux = UpSampling2D((2,2))(deconv2b_ux)
    unpool1_ux = MaxUnpooling2D(pool_size=(2, 2), out_shape=conv1b.shape)([deconv2b_ux, idx1])
    concat1_ux = Concatenate()([conv1b, unpool1_ux])
    deconv1a_ux = Conv2DTranspose(8, (5,5), activation='relu', padding='same')(concat1_ux)
    deconv1b_ux = Conv2DTranspose(1, (5,5), padding='same')(deconv1a_ux)

    # Separate Uy decoder channel
    # upsamp4_uy = UpSampling2D((2,2))(pool4)
    # unpool4_uy = MaxUnpooling2D(pool_size=(2, 2), out_shape=conv4b.shape)([pool4, idx4])
    concat4_uy = Concatenate()([conv4a, conv4b])
    deconv4a_uy = Conv2DTranspose(32, (5,5), activation='relu', padding='same')(concat4_uy)
    deconv4b_uy = Conv2DTranspose(32, (5,5), activation='relu', padding='same')(deconv4a_uy)

    # upsamp3_uy = UpSampling2D((2,2))(deconv4b_uy)
    unpool3_uy = MaxUnpooling2D(pool_size=(2, 2), out_shape=conv3b.shape)([deconv4b_uy, idx3])
    concat3_uy = Concatenate()([conv3b, unpool3_uy])
    deconv3a_uy = Conv2DTranspose(32, (5,5), activation='relu', padding='same')(concat3_uy)
    deconv3b_uy = Conv2DTranspose(16, (5,5), activation='relu', padding='same')(deconv3a_uy)

    # upsamp2_uy = UpSampling2D((2,2))(deconv3b_uy)
    unpool2_uy = MaxUnpooling2D(pool_size=(2, 2), out_shape=conv2b.shape)([deconv3b_uy, idx2])
    concat2_uy = Concatenate()([conv2b, unpool2_uy])
    deconv2a_uy = Conv2DTranspose(16, (5,5), activation='relu', padding='same')(concat2_uy)
    deconv2b_uy = Conv2DTranspose(8, (5,5), activation='relu', padding='same')(deconv2a_uy)

    # upsamp1_uy = UpSampling2D((2,2))(deconv2b_uy)
    unpool1_uy = MaxUnpooling2D(pool_size=(2, 2), out_shape=conv1b.shape)([deconv2b_uy, idx1])
    concat1_uy = Concatenate()([conv1b, unpool1_uy])
    deconv1a_uy = Conv2DTranspose(8, (5,5), activation='relu', padding='same')(concat1_uy)
    deconv1b_uy = Conv2DTranspose(1, (5,5), padding='same')(deconv1a_uy)
   
    # Separate p decoder channel
    # upsamp4_uy = UpSampling2D((2,2))(pool4)
    # unpool4_uy = MaxUnpooling2D(pool_size=(2, 2), out_shape=conv4b.shape)([pool4, idx4])
    concat4_p = Concatenate()([conv4a, conv4b])
    deconv4a_p = Conv2DTranspose(32, (5,5), activation='relu', padding='same')(concat4_p)
    deconv4b_p = Conv2DTranspose(32, (5,5), activation='relu', padding='same')(deconv4a_p)

    # upsamp3_uy = UpSampling2D((2,2))(deconv4b_uy)
    unpool3_p = MaxUnpooling2D(pool_size=(2, 2), out_shape=conv3b.shape)([deconv4b_p, idx3])
    concat3_p = Concatenate()([conv3b, unpool3_p])
    deconv3a_p = Conv2DTranspose(32, (5,5), activation='relu', padding='same')(concat3_p)
    deconv3b_p = Conv2DTranspose(16, (5,5), activation='relu', padding='same')(deconv3a_p)

    # upsamp2_uy = UpSampling2D((2,2))(deconv3b_uy)
    unpool2_p = MaxUnpooling2D(pool_size=(2, 2), out_shape=conv2b.shape)([deconv3b_p, idx2])
    concat2_p = Concatenate()([conv2b, unpool2_p])
    deconv2a_p = Conv2DTranspose(16, (5,5), activation='relu', padding='same')(concat2_p)
    deconv2b_p = Conv2DTranspose(8, (5,5), activation='relu', padding='same')(deconv2a_p)

    # upsamp1_uy = UpSampling2D((2,2))(deconv2b_uy)
    unpool1_p = MaxUnpooling2D(pool_size=(2, 2), out_shape=conv1b.shape)([deconv2b_p, idx1])
    concat1_p = Concatenate()([conv1b, unpool1_p])
    deconv1a_p = Conv2DTranspose(8, (5,5), activation='relu', padding='same')(concat1_p)
    deconv1b_p = Conv2DTranspose(1, (5,5), padding='same')(deconv1a_p)

    # # Creating Model
    model = Model(inputs=[inputs], outputs=[deconv1b_ux,deconv1b_uy,deconv1b_p])

    # Creating optimiser
    optimiser = AdamW(weight_decay, learning_rate)

    # creating metrics
    metrics = ['acc',]

    # creating separate losses
    losses = [MeanSquaredError(), MeanSquaredError(), MeanAbsoluteError()]

    # Compiling model
    model.compile(optimizer="adam", loss=losses, metrics=metrics)

    return model