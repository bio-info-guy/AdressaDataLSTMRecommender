
# coding: utf-8

# In[261]:

import json
import numpy as np
import matplotlib.pyplot as plt
#import pandas
import math
import sys
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import LSTM, GRU
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import keras.backend as K
from keras.callbacks import ModelCheckpoint
from keras.callbacks import Callback
from keras.callbacks import TensorBoard
from keras.regularizers import L1L2

# In[253]:


def batch_generator(x, t):
    i=0
    while True:
        if i == len(x):
            i=0
        else:
            xtrain, ytrain=x[i], t[i]
            i +=1
        yield xtrain, ytrain


# In[255]:


def LSTMbyTime(Xtrain, Ttrain,  Xtest, Ttest, epochs):
    K.clear_session()
    model = Sequential()
    model.add(LSTM(512, input_shape=(None, Xtrain[0].shape[2]), return_sequences=True, kernel_regularizer=L1L2(0.01, 0.01), dropout=0.5))
    model.add(LSTM(1024, input_shape=( None, Xtrain[0].shape[2]), kernel_regularizer=L1L2(0.01, 0.01),dropout=0.5))
    model.add(Dense(2048))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(Ttrain[0].shape[1]))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['cosine', 'mse'])
    model.summary()
    val_steps=len(Xtest)
    steps_per_epoch=len(Xtrain)
    filepath="time_weights.best.model"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    TB=TensorBoard(log_dir='./Graph_time', histogram_freq=0, write_graph=True, write_images=True)
    callbacks_list = [checkpoint, TB]
    history=model.fit_generator(batch_generator(Xtrain, Ttrain), steps_per_epoch=steps_per_epoch,epochs=epochs, validation_data=batch_generator(Xtest, Ttest), callbacks=callbacks_list, verbose=2, validation_steps=val_steps)
    return {'model':model, 'history':history}
    


# In[256]:


def LSTMbyFixSeq(Xtrain, Ttrain, batch_size, epochs):
    K.clear_session()
    filepath="fixed_weights.best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    TB=TensorBoard(log_dir='./Graph_fixed', histogram_freq=0,  
          write_graph=True, write_images=True)
    callbacks_list = [checkpoint, TB]
    model = Sequential()
    model.add(LSTM(512, input_shape=(Xtrain.shape[1], Xtrain.shape[2]), kernel_regularizer=L1L2(0.01, 0.01), dropout=0.1, return_sequences=True))
    model.add(LSTM(1024, input_shape=(Xtrain.shape[1], Xtrain.shape[2]), kernel_regularizer=L1L2(0.01, 0.01), dropout=0.5))
    model.add(Dense(2048))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(Ttrain.shape[1]))
    model.compile(loss='mean_squared_error' , optimizer='adam', metrics=['mse', 'cosine'])
    model.summary()
    history=model.fit(Xtrain, Ttrain,batch_size=batch_size,epochs=epochs, callbacks=callbacks_list, verbose=2, validation_split=0.2) 
    return {'model':model, 'history':history}


# In[258]:


def main(args):
    x_file=args[0]
    t_file=args[1]
    model=args[2]
    X=np.load(x_file)
    T=np.load(t_file)
    train_split=int(X.shape[0]//1.2)
    Xtrain, Ttrain=X[:train_split], T[:train_split]
    Xtest, Ttest=X[train_split:], T[train_split:]
    if model == 'time':
        test=LSTMbyTime(Xtrain, Ttrain, epochs=100)
    elif model == 'fixed':
        test=LSTMbyFixSeq(Xtrain, Ttrain, batch_size=50, epochs=100)
    with open(model+'history.json','w') as f:
        json.dump(test['history'].history, f)
    


# In[262]:


if __name__ == '__main__':
    main(sys.argv[1:])

