import os

from einops import rearrange
import _pickle as pickle
import numpy as np
import math
import h5py
import time
import json
from bayes_opt import BayesianOptimization

import tensorflow as tf
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

from src.model import build_model
import src.metrics as metrics
# from src.datasets import BikeDC
from src.evaluation import evaluate
from cache_utils import cache, read_cache

import sys
from os.path import abspath, join, dirname
sys.path.insert(0, join(abspath(dirname(__file__)), '../',"data"))
from prepareDataBike import load_data_Bike
# tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6144)])
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:  # Currently, memory growth needs to be the same across GPUs
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)  # Memory growth must be set before GPUs have been initialized

# parameters
DATAPATH = '../data' 
nb_epoch = 100  # number of epoch at training stage
# nb_epoch_cont = 150  # number of epoch at training (cont) stage
T = 48  # number of time intervals in one day
CACHEDATA = True  # cache data or NOT

len_closeness = len_c = 2  # length of closeness dependent sequence
len_period = len_p = 0  # length of peroid dependent sequence
len_trend = len_t = 1  # length of trend dependent sequence
# len_cpt = [[2,0,1]]
# batch_size = [16, 64]  # batch size
# lr = [0.0015, 0.00015]  # learning rate
# lstm = [350, 500]
# lstm_number = [2, 3]

nb_flow = 2  # there are two types of flows: new-flow and end-flow
# divide data into two subsets: Train & Test, 
days_test = 7*4
len_test = T*days_test
len_val = 2*len_test

map_height, map_width = 32, 16  # grid size

path_cache = os.path.join(DATAPATH, 'CACHE', '3D-CLoST')  # cache path

path_result = 'RET'
path_model = 'MODEL'
if os.path.isdir(path_result) is False:
    os.makedirs(path_result)
if os.path.isdir(path_model) is False:
    os.makedirs(path_model)
if CACHEDATA and os.path.isdir(path_cache) is False:
    os.makedirs(path_cache)

    # load data
print("loading data...")
fname = os.path.join(path_cache, 'BikeDC_C{}_P{}_T{}.h5'.format(
    len_closeness, len_period, len_trend))
def split_to_flow_ext_data(X_train_all):
    for i in range(len(X_train_all)):
        if len(X_train_all[i].shape) == 4:
            X_train_all[i] = rearrange(X_train_all[i], "n (c1 c) h w -> n c h w c1",c1=2)
    x_flow = np.concatenate(X_train_all[:-1],axis=1)
    x_ext = X_train_all[-1]
    res = [x_flow,x_ext]
    return res

if os.path.exists(fname) and CACHEDATA:
    X_train_all, Y_train_all, X_train, Y_train, X_val, Y_val, X_test, Y_test, mmn, external_dim, timestamp_train_all, timestamp_train, timestamp_val, timestamp_test, mask = read_cache(
        fname,"preprocessing_BikeDC.pkl")
    print("load %s successfully" % fname)
else:
    X_train_all, Y_train_all, X_test, Y_test, mmn, external_dim, timestamp_train_all, timestamp_test = load_data_Bike(T=T, nb_flow=nb_flow,dataset="BIKEDC201901-202201",
                      len_closeness=len_closeness, len_period=len_period, len_trend=len_trend,
                      len_test=len_test, meta_data=True, holiday_data=True, meteorol_data=True,prediction_offset=0)
    assert (len_closeness + len_period + len_trend > 0)
    X_train, Y_train = np.zeros((3,3)),np.zeros((3,3)) 
    X_val, Y_val,timestamp_train, timestamp_val = np.zeros((3,3)),np.zeros((3,3)),np.zeros((3,3)),np.zeros((3,3))
    mask = np.ones((map_height, map_width, nb_flow))

    X_train_all = split_to_flow_ext_data(X_train_all)    
    X_test = split_to_flow_ext_data(X_test)
    
    Y_train_all = rearrange(Y_train_all, "n c h w -> n h w c")
    Y_test = rearrange(Y_test, "n c h w -> n h w c")
    if CACHEDATA:
        cache(fname=fname, X_train_all=X_train_all, Y_train_all=Y_train_all,X_test=X_test, Y_test=Y_test,
                external_dim=external_dim, timestamp_train_all=timestamp_train_all, timestamp_test=timestamp_test,
                X_train=X_train, Y_train=Y_train,X_val=X_val, Y_val=Y_val,timestamp_train=timestamp_train, 
                timestamp_val= timestamp_val, mask=mask)


print("\n days (test): ", [v[:8] for v in timestamp_test[0::T]])
print('=' * 10)


def train_model(lr, batch_size, lstm, lstm_number, save_results=False, i=''):
    # get discrete parameters
    lstm = 350 if lstm < 1 else 500
    lstm_number = int(lstm_number)
    batch_size = 16 if batch_size < 2 else 64
    lr = round(lr,5)
    mask = np.ones((map_height, map_width, nb_flow))
    # build model
    model = build_model('NY', X_train_all,  Y_train_all, conv_filt=64, kernel_sz=(2,3,3), 
                    mask=mask, lstm=lstm, lstm_number=lstm_number, add_external_info=True,
                    lr=lr, save_model_pic=None)

    # model.summary()
    hyperparams_name = 'BikeDC{}.c{}.p{}.t{}.lstm_{}.lstmnumber_{}.lr_{}.batchsize_{}'.format(
            i, len_c, len_p, len_t, lstm, lstm_number, lr, batch_size)
    fname_param = os.path.join('MODEL', '{}.best.h5'.format(hyperparams_name))

    early_stopping = EarlyStopping(monitor='val_rmse', patience=25, mode='min')
    # lr_callback = LearningRateScheduler(lrschedule)
    model_checkpoint = ModelCheckpoint(
        fname_param, monitor='val_rmse', verbose=0, save_best_only=True, mode='min')

    # train model
    print("training model...")
    ts = time.time()
    if (i):
        print(f'Iteration {i}')
        np.random.seed(i * 18)
        tf.random.set_seed(i * 18)
    history = model.fit(X_train_all, Y_train_all,
                        epochs=100,
                        batch_size=batch_size,
                        validation_data=(X_test, Y_test),
                        callbacks=[model_checkpoint],
                        # callbacks=[model_checkpoint, lr_callback],
                        # callbacks=[model_checkpoint],
                        verbose=2)
    model.save_weights(os.path.join(
        'MODEL', '{}.h5'.format(hyperparams_name)), overwrite=True)
    # pickle.dump((history.history), open(os.path.join(
    #     path_result, '{}.history.pkl'.format(hyperparams_name)), 'wb'))
    print("\nelapsed time (training): %.3f seconds\n" % (time.time() - ts))

    # evaluate
    model.load_weights(fname_param)
    score = model.evaluate(
        X_test, Y_test, batch_size=Y_test.shape[0], verbose=0)
    print('Test score: %.6f rmse (norm): %.6f rmse (real): %.6f' %
          (score[0], score[1], score[1] * (mmn._max - mmn._min) / 2.))

    if (save_results):
        print('evaluating using the model that has the best loss on the valid set')
        model.load_weights(fname_param)  # load best weights for current iteration

        Y_pred = model.predict(X_test)  # compute predictions

        score = evaluate(Y_test, Y_pred, mmn, rmse_factor=1)  # evaluate performance

        # save to csv
        csv_name = os.path.join('results', '3dclost_BikeDC_results.csv')
        if not os.path.isfile(csv_name):
            if os.path.isdir('results') is False:
                os.makedirs('results')
            with open(csv_name, 'a', encoding="utf-8") as file:
                file.write('iteration,'
                           'rsme_in,rsme_out,rsme_tot,'
                           'mape_in,mape_out,mape_tot,'
                           'ape_in,ape_out,ape_tot'
                           )
                file.write("\n")
                file.close()
        with open(csv_name, 'a', encoding="utf-8") as file:
            file.write(f'{i},{score[0]},{score[1]},{score[2]},{score[3]},'
                       f'{score[4]},{score[5]},{score[6]},{score[7]},{score[8]}'
                       )
            file.write("\n")
            file.close()
        K.clear_session()

    # bayes opt is a maximization algorithm, to minimize validation_loss, return 1-this
    bayes_opt_score = 1.0 - score[1]

    return bayes_opt_score

# bayesian optimization
optimizer = BayesianOptimization(f=train_model,
                                 pbounds={'lstm': (0, 1.999), # 350 if smaller than 1 else 500
                                          'lstm_number': (2, 3.999),
                                          'lr': ( 0.0001,0.001),
                                          'batch_size': (1, 2.999), # 16 if smaller than 2 else 64
                                        #   'kernel_size': (3, 5.999)
                                 },
                                 verbose=2)

optimizer.maximize(init_points=2, n_iter=5)

# training-test-evaluation iterations with best params
if os.path.isdir('results') is False:
    os.makedirs('results')
targets = [e['target'] for e in optimizer.res]
bs_fname = 'bs_BikeDC.json'
with open(os.path.join('results', bs_fname), 'w') as f:
    json.dump(optimizer.res, f, indent=2)
best_index = targets.index(max(targets))
params = optimizer.res[best_index]['params']
# save best params
params_fname = '3dclost_BikeDC_best_params.json'
with open(os.path.join('results', params_fname), 'w') as f:
    json.dump(params, f, indent=2)
# with open(os.path.join('results', params_fname), 'r') as f:
#     params = json.load(f)
for i in range(0, 10):
    train_model(lstm=params['lstm'],
                lstm_number=params['lstm_number'],
                lr=params['lr'],
                batch_size=params['batch_size'],
                save_results=True,
                i=i)
