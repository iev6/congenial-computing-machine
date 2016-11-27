''''
Implementation of a CNN based Speaker Identification framework using keras

Dataset Used : TIMIT

@Author : Giridhur S. <giridhursriram@gmail.com>

''''



import cPickle as pkl
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import MiniBatchSparsePCA,PCA,IncrementalPCA
from sklearn.externals import joblib

from keras.models import Sequential
from keras.optimizers import SGD,Nadam
from keras.layers import Dense, Activation,Dropout
from keras.regularizers import l2, activity_l2
from keras.layers.advanced_activations import LeakyReLU

#We will be building on the MNIST example


model = Sequential()
model.add(Dense(2000, input_dim=3073, init='uniform'))#W_regularizer=l2(1e-5), activity_regularizer=activity_l2(1e-5)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1000, init='uniform'))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(12, init='uniform'))#W_regularizer=l2(1e-5), activity_regularizer=activity_l2(1e-5)))
model.add(Activation('softmax'))
optim = Nadam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-04, schedule_decay=0.0001)
optim = SGD(lr=0.1, momentum=0.1, decay=0.1, nesterov=False)
model.compile(loss='categorical_crossentropy',optimizer=optim,metrics=['accuracy'])
epochs = 200
batch_size = 256

model.fit(trX,trY,validation_data=(teX,teY),nb_epoch=epochs,batch_size=batch_size)
score = model.evaluate(teX,teY, batch_size=batch_size)
