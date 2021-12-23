from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, GlobalAveragePooling2D, Dense, Activation, LeakyReLU
from tensorflow.keras.applications.vgg19 import VGG19

def generator_network(image_shape):

    input_tensor = Input(shape=image_shape)

    x = Conv2D(64, 3, strides=1, padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = Conv2D(128, 3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = Conv2D(256, 3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = Conv2D(128, 3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = Conv2D(64, 3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = Conv2D(3, 3, strides=1, padding='same')(x)
    output_tensor = Activation('tanh')(x)

    return(Model(inputs=input_tensor, outputs=output_tensor))

def discriminator_network(image_shape):

    input_tensor = Input(shape=image_shape)

    x = Conv2D(64, 3, strides=1, padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = Conv2D(128, 3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = Conv2D(256, 3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = Conv2D(512, 3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = GlobalAveragePooling2D()(x)
    output_tensor = Dense(1, activation='sigmoid')(x) 

    return(Model(inputs=input_tensor, outputs=output_tensor))

def end_to_end_gan(discriminator_network, generator_network, image_shape):
    discriminator_network.trainable = False
    input_tensor = Input(shape=image_shape)
    generator_output = generator_network(input_tensor)
    discriminator_output = discriminator_network(generator_output)
    return(Model(inputs=input_tensor, outputs=[generator_output, discriminator_output]))

def content_loss(gt, pred):
    vgg_net = VGG19(include_top=False, weights='imagenet', input_shape=(250,250,3))
    embedding_model = Model(inputs=vgg_net.input, outputs=vgg_net.get_layer('block5_conv4').output)
    embedding_model.trainable = False
    loss = K.mean(K.square(embedding_model(gt) - embedding_model(pred)))
    return(loss)
