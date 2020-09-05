# u-net model with up-convolution or up-sampling and weighted binary-crossentropy as loss func

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Activation
from keras.optimizers import Adam
from keras.utils import plot_model
from keras import backend as K

def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal", padding="same")(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal", padding="same")(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

def unet_model(n_classes=5, im_sz=160, n_channels=8, n_filters_start=32, growth_factor=2, upconv=True,
               class_weights=[0.2, 0.3, 0.1, 0.1, 0.3], dropout=0, batchnorm=True):
    n_filters = n_filters_start
    inputs = Input((im_sz, im_sz, n_channels))
    conv1 = conv2d_block(inputs, n_filters, kernel_size=3, batchnorm=batchnorm)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1 = Dropout(dropout*0.5)(pool1)

    n_filters *= growth_factor
    conv2 = conv2d_block(pool1, n_filters, kernel_size=3, batchnorm=batchnorm)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2 = Dropout(dropout)(pool2)

    n_filters *= growth_factor
    conv3 = conv2d_block(pool2, n_filters, kernel_size=3, batchnorm=batchnorm)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    pool3 = Dropout(dropout)(pool3)

    n_filters *= growth_factor
    conv4 = conv2d_block(pool3, n_filters, kernel_size=3, batchnorm=batchnorm)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    pool4 = Dropout(dropout)(pool4)

    n_filters *= growth_factor
    conv5 = conv2d_block(pool4, n_filters, kernel_size=3, batchnorm=batchnorm)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)
    pool5 = Dropout(dropout)(pool5)

    n_filters *= growth_factor
    conv6 = conv2d_block(pool5, n_filters, kernel_size=3, batchnorm=batchnorm)

    n_filters //= growth_factor
    if upconv:
        up7 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(conv6), conv5])
    else:
        up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv5])
    up7 = Dropout(dropout)(up7)
    conv7 = conv2d_block(up7, n_filters, kernel_size=3, batchnorm=batchnorm)

    n_filters //= growth_factor
    if upconv:
        up8 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(conv7), conv4])
    else:
        up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv4])
    up8 = Dropout(dropout)(up8)
    conv8 = conv2d_block(up8, n_filters, kernel_size=3, batchnorm=batchnorm)

    n_filters //= growth_factor
    if upconv:
        up9 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(conv8), conv3])
    else:
        up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv3])
    up9 = Dropout(dropout)(up9)
    conv9 = conv2d_block(up9, n_filters, kernel_size=3, batchnorm=batchnorm)

    n_filters //= growth_factor
    if upconv:
        up10 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(conv9), conv2])
    else:
        up10 = concatenate([UpSampling2D(size=(2, 2))(conv9), conv2])
    up10 = Dropout(dropout)(up10)
    conv10 = conv2d_block(up10, n_filters, kernel_size=3, batchnorm=batchnorm)

    n_filters //= growth_factor
    if upconv:
        up11 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(conv10), conv1])
    else:
        up11 = concatenate([UpSampling2D(size=(2, 2))(conv10), conv1])
    up11 = Dropout(dropout*0.5)(up11)
    conv11 = conv2d_block(up11, n_filters, kernel_size=3, batchnorm=batchnorm)

    conv12 = Conv2D(n_classes, (1, 1), activation='sigmoid')(conv11)

    model = Model(inputs=inputs, outputs=conv12)

    def weighted_binary_crossentropy(y_true, y_pred):
        class_loglosses = K.mean(K.binary_crossentropy(y_true, y_pred), axis=[0, 1, 2])
        return K.sum(class_loglosses * K.constant(class_weights))

    model.compile(optimizer=Adam(), loss=weighted_binary_crossentropy)
    return model


if __name__ == '__main__':
    model = unet_model()
    print(model.summary())
    plot_model(model, to_file='unet_model.png', show_shapes=True)
