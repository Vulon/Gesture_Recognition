import pandas as pd
import cv2
import numpy as np
import os

default_width = 176
default_height = 144
default_channels = 3


def readTrainFile(path):
    frame = pd.read_csv(path, header=0)
    fileList = frame['Path'].values.tolist()
    dataList = frame[['Object_x1', 'Object_y1', 'Object_x2', 'Object_y2']].to_numpy()
    return fileList, dataList

def prepareImageData(fileList):
    l = list(fileList)
    a = []
    for x in range(len(l)):
        image = cv2.imread(l[x])
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image,(default_width, default_height),interpolation=cv2.INTER_AREA)
        image = np.reshape(a=image, newshape=(default_height, default_width, default_channels))
        a.append(image)
    a = np.asarray(a)
    a.reshape((-1, default_height, default_width, default_channels))
    a = a.astype('float64')
    a = a / 255.0
    return a

def prepareLabelData(dataList):
    dataList = dataList.astype('float64')
    for x in dataList[:]:
        x[0] = max(x[0], 0)
        x[0] = min(x[0], default_width)
        x[2] = max(x[2], 0)
        x[2] = min(x[2], default_width)
        x[1] = max(x[1], 0)
        x[1] = min(x[1], default_height)
        x[3] = max(x[3], 0)
        x[3] = min(x[3], default_height)
        x[0] = x[0] / default_width
        x[2] = x[2] / default_width
        x[1] = x[1] / default_height
        x[3] = x[3] / default_height


    return dataList

def getFitData(trainFile=r'D:/Database/train1.csv'):
    X, Y = readTrainFile(trainFile)
    X = prepareImageData(X)
    Y = prepareLabelData(Y)
    return X, Y

def restoreLabelData(dataList):
    for x in dataList[:]:
        x[0] *= default_width
        x[1] *= default_height
        x[2] *= default_width
        x[3] *= default_height

    return dataList

def readTestFiles(path=r'D:\Database\test'):
    names = []
    for dirname, _, filenames in os.walk(path):
        for filename in filenames:
            name = os.path.join(dirname, filename)
            names.append(name)

    return names

def drawRectangle(image, x1, y1, x2, y2):
    x1 = max(x1, 0)
    x1 = min(x1, default_width)
    x2 = max(x2, 0)
    x2 = min(x2, default_width)
    y1 = max(y1, 0)
    y1 = min(y1, default_height)
    y2 = max(y2, 0)
    y2 = min(y2, default_height)
    cv2.rectangle(image, (x1, y1), (x2, y2),(0, 0, 255), 1)

def show_image_with_rectangle(path, label, i, test_name):
    image = cv2.imread(path)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (default_width, default_height), interpolation=cv2.INTER_AREA)
    drawRectangle(image, label[0], label[1], label[2], label[3])
    print(label)
    cv2.imshow('image', image)
    cv2.imwrite('saved_data/images/' + test_name + '_' + str(i) + '.png', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()