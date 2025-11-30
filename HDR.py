import numpy as np
import idx2numpy

def data_handling(pathimages, pathlabels):
    images = idx2numpy.convert_from_file(pathimages).reshape((60000, 784))
    images = images / 255.0
    labels = idx2numpy.convert_from_file(pathlabels)
    map = np.identity(10)
    labels_encoded = map[labels] #one-hot encoding
    return images, labels_encoded



pathimagess = 'dataset/train-images.idx3-ubyte'
pathlabelss = 'dataset/train-labels.idx1-ubyte'
data_handling(pathimagess, pathlabelss)

