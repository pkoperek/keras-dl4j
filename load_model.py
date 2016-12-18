from __future__ import print_function
import numpy as np

np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import load_model
from keras import backend as K
from keras.utils import np_utils

# input image dimensions
img_rows, img_cols = 28, 28
# no of classes
nb_classes = 10

(X_train, y_train), (X_test, y_test) = mnist.load_data()
if K.image_dim_ordering() == 'th':
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
else:
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)

X_test = X_test.astype('float32')
X_test /= 255

Y_test = np_utils.to_categorical(y_test, nb_classes)

model = load_model("/tmp/mnist_model.hd5")
model.load_weights("/tmp/mnist_weights_dl4j.hd5")

score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])