''''
Implementation of a CNN based Speaker Identification framework using keras

Dataset Used : TIMIT <stored in train_data.pkl>

@Authors : Giridhur S. <giridhursriram@gmail.com>, Ramasubramanian B. <brama1995@gmail.com>

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
from keras.utils.io_utils import HDF5Matrix
#We will be building on the MNIST example

#with open('train_data.pkl','rb') as f1:
#    train = pkl.load(f1)



batch_size = 32
nb_classes = np.max(label)+1 #numerically equal to len(speaker.keys())
nb_epoch = 50

# input image dimensions
img_rows, img_cols = 256, 100

train = HDF5Matrix('train_data.h5','imgs');
label = HDF5Matrix('train_data.h5','lbls');
#train = train[:,:-1]
batch_size = 32
nb_classes = np.max(label)+1 #numerically equal to len(speaker.keys())
nb_epoch = 50

train_label = np_utils.to_categorical(label,nb_classes=np.int32(nb_classes)) #HACK : nb_classes hardcoded!!!
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
                        input_shape=input_shape,init='glorot_normal'))
model.add(Activation('relu'))
#L2
model.add(MaxPooling2D(pool_size=pool_size))
#L3
model.add(Convolution2D(nb_filters2,kernel_size[0], kernel_size[1],subsample=stride,init='glorot_normal'))
model.add(Activation('relu'))
#L4
model.add(MaxPooling2D(pool_size=pool_size))

#L5
#model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(n_s*10,init='glorot_normal'))
model.add(Activation('relu'))

#L6
model.add(Dropout(0.5))

#L7
model.add(Dense(5*n_s,init='glorot_normal'))
model.add(Activation('relu'))

#L8
model.add(Dense(n_s))
model.add(Activation('softmax'))
#Optimizer
optim = Nadam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-06, schedule_decay=0.0001)
model.compile(loss='categorical_crossentropy',optimizer=optim,metrics=['accuracy'])

#Training
model.fit(trX, trY, batch_size=batch_size, nb_epoch=100,
          verbose=1, validation_data=(teX, teY),shuffle=False) ##NOTE
#Testing
score = model.evaluate(X_test, Y_test, verbose=0)
