from keras.layers import Input, Flatten, Dense, ELU, BatchNormalization, Conv2D, MaxPooling2D
from keras.models import Model, Sequential
from keras.callbacks import  EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from keras import optimizers
from keras.models import load_model

test_name = 'dense_model_05_075'
batch_size = 20
epochs = 350

import preprocess
import layers







class HandDetectionNetwork:
    def __init__(self):

        self.model = layers.create_dense_model(batch_size)


    def fit(self,epo, path=r'D:/Database/train1.csv'):
        X, Y = preprocess.getFitData(path)
        es = EarlyStopping(patience=epochs/ 2, restore_best_weights=True)
        #re = ReduceLROnPlateau(factor=0.5, patience=10, min_lr=0.001)
        history = self.model.fit(X, Y, batch_size=batch_size, epochs=epo,
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
            preprocess.show_image_with_rectangle(names[x], results[x], x, test_name)



net = HandDetectionNetwork()

def load_my_model(name, iteration):
    #name + str(iteration)
    net.model = load_model('saved_data/' + name + str(iteration) + ' model ' + '.h5',
                           custom_objects={'loss_v2': layers.create_loss(0.5, 0.75)})
    net.model.load_weights('saved_data/' + name + str(iteration) + '.h5')

createNewModel = False
if createNewModel:
    net.fit(epochs)
    net.model.save_weights('saved_data/' + test_name + '.h5')
    net.model.save('saved_data/' + test_name + ' model ' + '.h5')
else:
    load_my_model(test_name, 2)
    for i in range(3, 10):
        print("Started ", i, 'iteration')
        net.fit(100)
        net.model.save_weights('saved_data/' + test_name + str(i) + '.h5')
        net.model.save('saved_data/' + test_name + str(i) + ' model ' + '.h5')
        print("Finished", i, 'iteration')
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

