from keras.layers import Input, Flatten, Dense, ELU, BatchNormalization, Conv2D, MaxPooling2D
from keras.models import Model, Sequential
from keras.callbacks import  EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from keras import optimizers

import preprocess
import layers


batch_size = 8
epochs = 100




class HandDetectionNetwork:
    def __init__(self):

        self.model = layers.create_dense_model(batch_size)


    def fit(self, path=r'D:/Database/train1.csv'):
        X, Y = preprocess.getFitData(path)
        es = EarlyStopping(patience=epochs, restore_best_weights=True)
        #re = ReduceLROnPlateau(factor=0.5, patience=10, min_lr=0.001)
        history = self.model.fit(X, Y, batch_size=batch_size, epochs=epochs,
                       validation_split=0.05, verbose=2, callbacks=[es])

        fig, ax = plt.subplots(2, 1)
        ax[0].plot(history.history['loss'], color='b', label="Training loss")
        ax[0].plot(history.history['val_loss'], color='r', label="validation loss", axes=ax[0])
        legend = ax[0].legend(loc='best', shadow=True)


        res = self.model.predict(X[:3])
        for i in range(3):
            layers.show_loss(Y[i], res[i])



    def predict(self, pathList):
        X = preprocess.prepareImageData(pathList)
        results = self.model.predict(X)
        print("results: ", results)
        results = preprocess.restoreLabelData(results)
        print("restored results: ", results)
        return results

    def show(self):
        names = preprocess.readTestFiles()
        images = preprocess.prepareImageData(names)
        results = self.model.predict(images)
        print('results: ', results[0])
        results = preprocess.restoreLabelData(results)
        print('restored results: ', results[0])
        for x in range(len(names)):
            preprocess.show_image_with_rectangle(names[x], results[x])





net = HandDetectionNetwork()
createNewModel = True
if createNewModel:
    net.fit()
    net.model.save_weights('dense_model.h5')
else:
    net.model.load_weights('dense_model.h5')

net.show()
#test_names = preprocess.readTestFiles()
#net.predict(test_names)





        # act = layers.act
        # inp = Input((preprocess.default_height, preprocess.default_width, 1))
        # res1 = layers.res_block(inp, filter=8, unit_count=3)
        # filterUp1 = layers.Conv2D(kernel_size=1, filters=16, padding='same')(res1)
        # pool1 = MaxPooling2D()(filterUp1)
        # res2 = layers.res_block(pool1, filter=16, unit_count=3)
        # filterUp2 = layers.Conv2D(kernel_size=1, filters=24, padding='same')(res2)
        # pool2 = MaxPooling2D()(filterUp2)
        # res3 = layers.res_block(pool2, filter=24, unit_count=2)
        # flat = Flatten()(res3)
        # cn1 = Dense(units=400, activation=act)(flat)
        # cn2 = Dense(units=100, activation=act)(cn1)
        # cn3 = Dense(units=4, activation='sigmoid')(cn2)
        # self.model = Model(inputs=inp, outputs=cn3)
        # print(self.model.summary())
        # optimizer = optimizers.SGD(lr=0.02, momentum=0.01)
        #
        # self.model.compile(optimizer=optimizer, loss=layers.create_loss(batch_size), metrics=['accuracy'])

