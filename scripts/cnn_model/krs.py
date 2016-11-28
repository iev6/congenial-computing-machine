''''
Implementation of a CNN based Speaker Identification framework using keras

Dataset Used : TIMIT

@Author : Giridhur S. <giridhursriram@gmail.com>

''''



import cPickle as pkl
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Nadam
from sklearn.cross_validation import train_test_split
#We will be building on the MNIST example

with open('train_data.pkl','rb') as f1:
    train = pkl.load(f1)
with open('test_data.pkl','rb') as f1:
    test = pkl.load(f1)



batch_size = 128
nb_classes = 38
nb_epoch = 50

# input image dimensions
img_rows, img_cols = 128, 100

train_label = np_utils.to_categorical(train[:,-1]-1,nb_classes=38) #HACK : nb_classes hardcoded!!!
train = train[:,:-1]
train = train - np.mean(train,axis=0)
(trX,teX,trY,teY) = train_test_split(train,train_label,test_size=0.2)
trX = trX.reshape(trX.shape[0],img_rows,img_cols,1)
teX = teX.reshape(teX.shape[0],img_rows,img_cols,1)

# number of convolutional filters to use
n_s = nb_classes
nb_filters1 = 64
nb_filters2 = 128
# size of pooling area for max pooling
pool_size = (4, 4)
# convolution kernel size
kernel_size = (5, 5) ##EDIT
stride = (2,2) #2 steps in both directions
input_shape = (img_rows, img_cols, 1)
model = Sequential()
#L1
model.add(Convolution2D(nb_filters1, kernel_size[0], kernel_size[1],subsample=stride,border_mode='valid',
                        input_shape=input_shape))
model.add(Activation('relu'))
#L2
model.add(MaxPooling2D(pool_size=pool_size))
#L3
model.add(Convolution2D(nb_filters2,kernel_size[0], kernel_size[1],subsample=stride))
model.add(Activation('relu'))
#L4
model.add(MaxPooling2D(pool_size=pool_size))

#L5
#model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(n_s*10))
model.add(Activation('relu'))

#L6
model.add(Dropout(0.5))

#L7
model.add(Dense(5*n_s))
model.add(Activation('relu'))

#L8
model.add(Dense(n_s))
model.add(Activation('softmax'))
#Optimizer
optim = Nadam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-06, schedule_decay=0.0001)
model.compile(loss='categorical_crossentropy',optimizer=optim,metrics=['accuracy'])

#Training
model.fit(trX, trY, batch_size=128, nb_epoch=100,
          verbose=1, validation_data=(teX, teY))
#Testing
score = model.evaluate(X_test, Y_test, verbose=0)
