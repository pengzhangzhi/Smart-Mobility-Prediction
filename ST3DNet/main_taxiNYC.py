from ST3DNet import *
import pickle
from utils import *
import os
import math
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.utils.vis_utils import plot_model
from keras.optimizers import Adam
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import ParameterGrid

from evaluation import evaluate

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

nb_epoch = 150  # number of epoch at training stage
batch_size = [16,32,64]  # batch size
T = 24  # number of time intervals in one day
lr = [0.00015, 0.00035]  # learning rate
# lr = 0.00002  # learning rate
len_closeness = 6  # length of closeness dependent sequence
len_period = 0  # length of peroid dependent sequence
len_trend = 4  # length of trend dependent sequence
nb_residual_unit = [4,5,6]   # number of residual units
nb_flow = 2  # there are two types of flows: new-flow and end-flow
days_test = 7*4  
len_test = T * days_test
map_height, map_width = 16, 8  # grid size

path_result = 'RET'
path_model = 'MODEL'
if os.path.isdir(path_result) is False:
    os.mkdir(path_result)
if os.path.isdir(path_model) is False:
    os.mkdir(path_model)

filename = os.path.join("../data", 'CACHE', 'ST3DNet', 'TaxiNYC_c%d_p%d_t%d_noext'%(len_closeness, len_period, len_trend))
f = open(filename, 'rb')
X_train = pickle.load(f)
Y_train = pickle.load(f)
X_test = pickle.load(f)
Y_test = pickle.load(f)
mmn = pickle.load(f)
external_dim = pickle.load(f)
timestamp_train = pickle.load(f)
timestamp_test = pickle.load(f)

for i in X_train:
    print(i.shape)

Y_train = mmn.inverse_transform(Y_train)  # X is MaxMinNormalized, Y is real value
Y_test = mmn.inverse_transform(Y_test)

c_conf = (len_closeness, nb_flow, map_height,
              map_width) if len_closeness > 0 else None
t_conf = (len_trend, nb_flow, map_height,
          map_width) if len_trend > 0 else None


# build grid for grid search
params = {
    'batch_size': batch_size,
    'lr': lr,
    'nb_residual_unit': nb_residual_unit
}

grid = ParameterGrid(params)
# print(grid)

for i in range(len(grid)):
    # extract current grid params
    nb_res_unit = grid[i]['nb_residual_unit']
    lr = grid[i]['lr']
    batch_size = grid[i]['batch_size']

    print('Step [{}/{}], res_unit {}, lr {}, batch_size {}'
        .format(i+1, len(grid), nb_res_unit, lr, batch_size))

    iterations = 1
    for iteration in range(0, iterations):
        # build model
        print(f'Iteration {iteration}')

        model = ST3DNet(c_conf=c_conf, t_conf=t_conf, external_dim=external_dim,
                        nb_residual_unit=nb_res_unit)
        adam = Adam(lr=lr)
        model.compile(loss='mse', optimizer=adam, metrics=[rmse])

        hyperparams_name = 'TaxiNYC.c{}.p{}.t{}.resunit_{}.lr_{}.batchsize_{}.iter{}'.format(
            len_closeness, len_period, len_trend, nb_res_unit, lr, batch_size, iteration)
        fname_param = os.path.join(path_model, '{}.best.h5'.format(hyperparams_name))
        print(hyperparams_name)

        early_stopping = EarlyStopping(monitor='val_rmse', patience=25, mode='min')
        model_checkpoint = ModelCheckpoint(
            fname_param, monitor='val_rmse', verbose=0, save_best_only=True, mode='min')

        print('=' * 10)
        # train model
        np.random.seed(i*18)
        tf.random.set_seed(i*18)
        print("training model...")
        history = model.fit(X_train, Y_train,
                            epochs=nb_epoch,
                            batch_size=batch_size,
                            validation_split=0.1,
                            callbacks=[early_stopping, model_checkpoint],
                            verbose=0)
        model.save_weights(os.path.join(
            path_model, '{}.h5'.format(hyperparams_name)), overwrite=True)
        pickle.dump((history.history), open(os.path.join(
            path_result, '{}.history.pkl'.format(hyperparams_name)), 'wb'))

        print('=' * 10)

        # evaluate model
        print('evaluating using the model that has the best loss on the valid set')
        model.load_weights(fname_param) # load best weights for current iteration
        
        Y_pred = model.predict(X_test) # compute predictions

        score = evaluate(Y_test, Y_pred, rmse_factor=1) # evaluate performance

        # save to csv
        csv_name = os.path.join('results','st3dnet_taxiNYC_results.csv')
        if not os.path.isfile(csv_name):
            if os.path.isdir('results') is False:
                os.mkdir('results')
            with open(csv_name, 'a', encoding = "utf-8") as file:
                file.write(
                  'nb_residual_unit,'
                  'learning rate,'
                  'batch_size,'
                  'iteration,'
                  'rsme_in,rsme_out,rsme_tot,'
                  'mape_in,mape_out,mape_tot,'
                  'ape_in,ape_out,ape_tot'
                )
                file.write("\n")
                file.close()
        with open(csv_name, 'a', encoding = "utf-8") as file:
            file.write(
              f'{nb_res_unit},'
              f'{lr},'
              f'{batch_size},'
              f'{iteration},'
              f'{score[0]},{score[1]},{score[2]},'
              f'{score[3]},{score[4]},{score[5]},'
              f'{score[6]},{score[7]},{score[8]}'
            )
            file.write("\n")
            file.close()
        K.clear_session()