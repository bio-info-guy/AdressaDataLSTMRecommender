
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
from keras.callbacks import CSVLogger
from keras.callbacks import Callback

from keras.regularizers import l2 as L2
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform, conditional

def batch_generator(x, t):
    i=0
    while True:
        if i == len(x):
            i=0
        else:
            xtrain, ytrain=x[i], t[i]
            i +=1
    yield xtrain, ytrain

def create_time_model(Xtrain, Ttrain,   Xtest, Ttest):
    def batch_generator(x, t):
        i=0
        while True:
            if i == len(x):
                i=0
            else:
                xtrain, ytrain=x[i], t[i]
                i +=1
            yield xtrain, ytrain
    steps_per_epoch=len(Xtrain)
    val_steps=len(Xtest)
    model = Sequential()
    csv_logger = CSVLogger('time_log.csv', append=True, separator='\t')
    layer=conditional({{choice(['one', 'two'])}})
    if layer == 'two':
        returnseq=True
    else:
        returnseq=False
    model.add(LSTM(units={{choice([32, 64, 128, 256])}}, input_shape=(None, Xtrain[0].shape[2]), kernel_regularizer=L2({{uniform(0, 1)}}), dropout={{uniform(0,1)}}, return_sequences=returnseq))
    if layer == 'two':
        model.add(LSTM(units={{choice([256,512])}}, input_shape=(None, Xtrain[0].shape[2]), kernel_regularizer=L2({{uniform(0,1)}}), dropout={{uniform(0,1)}}))
    model.add(Dense({{choice([1024, 512])}}))
    model.add(Activation('relu'))
    model.add({{choice([Dropout(0.5), Activation('linear')])}})
    model.add(Dense(Ttrain[0].shape[1]))
    model.compile(loss='mean_squared_error' , optimizer={{choice(['rmsprop', 'adam', 'sgd'])}}, metrics=['cosine'])
    model.summary()
    history=model.fit_generator(batch_generator(Xtrain, Ttrain), steps_per_epoch=len(Xtrain),epochs=5, callbacks=[csv_logger], verbose=2, validation_data=batch_generator(Xtest, Ttest), validation_steps=len(Xtest))
    score, acc = model.evaluate_generator(batch_generator(Xtest, Ttest), steps=len(Xtest))
    return {'loss': acc, 'model':model, 'status': STATUS_OK}
    


# In[256]:


def create_fix_model(Xtrain, Ttrain, Xtest, Ttest):
    csv_logger = CSVLogger('fix_log.csv', append=True, separator='\t')
    model = Sequential()
    layer=conditional({{choice(['one', 'two'])}})
    if layer == 'two':
        returnseq=True
    else:
        returnseq=False
    model.add(LSTM(units={{choice([32, 64, 128, 256])}}, input_shape=(Xtrain.shape[1], Xtrain.shape[2]), kernel_regularizer=L2({{uniform(0, 1)}}), dropout={{uniform(0,1)}}, return_sequences=returnseq))
    if layer == 'two':
        model.add(LSTM(units={{choice([256,512])}}, input_shape=(Xtrain.shape[1], Xtrain.shape[2]), kernel_regularizer=L2({{uniform(0,1)}}), dropout={{uniform(0,1)}}))
    model.add(Dense({{choice([1024, 512])}}))
    model.add(Activation('relu'))
    model.add({{choice([Dropout(0.5), Activation('linear')])}})
    model.add(Dense(Ttrain.shape[1]))
    model.compile(loss='mean_squared_error' , optimizer={{choice(['rmsprop', 'adam', 'sgd'])}}, metrics=['cosine'])
    model.summary()
    model.fit(Xtrain, Ttrain,batch_size=50,epochs=5, callbacks=[csv_logger], verbose=2, validation_data=(Xtest, Ttest))
    score, acc = model.evaluate(Xtest, Ttest, verbose=0)
    return {'loss': acc, 'model':model, 'status': STATUS_OK}


# In[258]:

def data():        
    with open('temp_arg.json') as f:
        args=json.load(f)
    x_file, t_file, xcold_file, tcold_file = args['Xf'], args['Tf'], args['Xcold'], args['Tcold']
    Xtrain=np.load(x_file)
    Ttrain=np.load(t_file)
    Xtest=np.load(xcold_file)
    Ttest=np.load(tcold_file)
    return Xtrain, Ttrain, Xtest, Ttest


def main(args):
    x_file=args[0]
    t_file=args[1]
    xcold_file=args[2]
    tcold_file=args[3]
    argdict={'Xf':x_file, 'Tf':t_file, 'Xcold':xcold_file,'Tcold':tcold_file}
    with open('temp_arg.json', 'w') as f:
        json.dump(argdict, f)
    model=args[4]
    if model == 'time':
        best_run, best_model = optim.minimize(model=create_time_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=50,
                                          trials=Trials())
        X_train, Y_train, X_test, Y_test = data()
        print("Evalutation of best performing model:")
        print(best_model.evaluate_generator(batch_generator(X_test, Y_test), steps=len(X_test)))
        print("Best performing model chosen hyper-parameters:")
        print(best_run) 
    elif model == 'fixed':
        best_run, best_model = optim.minimize(model=create_fix_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=50,
                                          trials=Trials())
        X_train, Y_train, X_test, Y_test = data()
        print("Evalutation of best performing model:")
        print(best_model.evaluate(X_test, Y_test))
        print("Best performing model chosen hyper-parameters:")
        print(best_run)



if __name__ == '__main__':
    main(sys.argv[1:])

