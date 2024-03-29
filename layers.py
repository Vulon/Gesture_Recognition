from keras.layers import Conv2D, ELU, BatchNormalization, concatenate, add, LeakyReLU, Input, MaxPooling2D, Flatten
import numpy as np
from keras.layers import Dense, Dropout
from keras.models import Model
from keras import optimizers
import keras.backend as K



act = 'sigmoid'

from preprocess import default_height, default_width, default_channels

#act = LeakyReLU()



def __resUnit(inputs, filters):
    x = BatchNormalization()(inputs)
    x = Conv2D(filters=filters, kernel_size=(1, 3), padding='same', activation=act)(x)
    x = Conv2D(filters=filters, kernel_size=(3, 1), padding='same', activation=act)(x)
    #x = ELU()(x)
    return x

def res_block(x, filter, unit_count):
    inputs = x
    for i in range(unit_count):
        x = __resUnit(x, filter)
    x = add([inputs, x], )
    #x = ELU()(x)
    return x

def create_dense_model(batch_size):
    act = 'sigmoid'
    inp = Input((default_height, default_width, default_channels))
    x = dense_block(inp, conv_filters=8, unit_count=3, activation=act)
    x = dense_block(x, conv_filters=6, unit_count=3, activation=act)
    x = dense_block(x, conv_filters=4, unit_count=3, activation=act)
    x = dense_block(x, conv_filters=4, unit_count=3, activation=act)
    x = Flatten()(x)
    x = Dense(units=512, activation=act)(x)
    x = Dropout(0.25)(x)
    x = Dense(128, activation=act)(x)
    x = Dense(4, activation=act)(x)
    model = Model(inp, x)
    optimizer = optimizers.SGD(lr=0.05, momentum=0.01)

    model.compile(optimizer=optimizer, loss=create_loss(error_alpha=0.5, size_alpha=0.75))
    print(model.summary())
    return model


def __denseUnit(inputs, filters, activation):
    x = BatchNormalization()(inputs)
    #x = Conv2D(filters, kernel_size=(1, 3), padding='same')(x)
    #x = Conv2D(filters, kernel_size=(3, 1), padding='same', activation=activation)(x)
    x = Conv2D(filters, kernel_size=3, padding='same', activation=activation)(x)

    return x

def dense_block(inputs, unit_count, conv_filters, activation):
    concat_inputs = inputs
    for i in range(unit_count):
        x = __denseUnit(concat_inputs, conv_filters, activation=activation)
        concat_inputs = concatenate([concat_inputs, x], axis=-1)
    concat_inputs = MaxPooling2D()(concat_inputs)

    return concat_inputs

def create_loss(error_alpha, size_alpha):

    def loss_v2(y_true, y_pred):
        maximum = K.maximum(y_true, y_pred)
        minimum = K.minimum(y_true, y_pred)
        #zero = K.zeros_like(maximum[:, 0])
        inter_area = K.maximum(0.0, minimum[:, 2] - maximum[:, 0] + K.epsilon()) * K.maximum(0.0, minimum[:, 3] - maximum[:, 1] + K.epsilon())
        true_area = (y_true[:, 2] - y_true[:, 0] + K.epsilon()) * (y_true[:, 3] - y_true[:, 1] + K.epsilon())
        pred_area = (y_pred[:, 2] - y_pred[:, 0] + K.epsilon()) * (y_pred[:, 3] - y_pred[:, 1] + K.epsilon())

        error1 = K.abs((y_pred[:, 2] - y_pred[:, 0]) - K.abs(y_pred[:, 2] - y_pred[:, 0]))
        error2 = K.abs((y_pred[:, 3] - y_pred[:, 1]) - K.abs(y_pred[:, 3] - y_pred[:, 1]))

        res = 1 - (inter_area / (true_area + pred_area - inter_area)) + (error1 + error2) * error_alpha
        res = res + K.maximum(0.0, (true_area - pred_area) * size_alpha)
        return K.mean(res)

    return loss_v2

def show_loss(y_true, y_pred):
    xA = max(y_true[0], y_pred[0])
    yA = max(y_true[1], y_pred[1])
    xB = min(y_true[2], y_pred[2])
    yB = min(y_true[3], y_pred[3])
    inter_area = max(0, xB - xA) * max(0, yB - yA)
    area1 = (y_true[3] - y_true[1]) * (y_true[2] - y_true[0])
    area2 = (y_pred[3] - y_pred[1]) * (y_pred[2] - y_pred[0])
    print("True: ", y_true)
    print("Pred: ", y_pred)
    print("inter area", inter_area)
    print("Union area", area1 + area2 - inter_area)
    print("result ", 1 - (inter_area / (area2 + area1 - inter_area)))

