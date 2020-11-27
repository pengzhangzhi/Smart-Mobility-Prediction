import numpy as np
import time
import math
import os
import pickle as pickle
from keras.callbacks import EarlyStopping, ModelCheckpoint
from bayes_opt import BayesianOptimization
import tensorflow as tf
from keras import backend as K

from utils import cache, read_cache
from src import BikeNYC, BikeNYC3d
from src.evaluation import evaluate
from src import (
    model as m1,
    model2 as m2,
    model3 as m3,
    model4 as m4,
    model5 as m5
)

models_dict = {
    'model1': m1,
    'model2': m2,
    'model3': m3,
    'model4': m4,
    'model5': m5,
}

# parameters
model_name = 'model2'

DATAPATH = '../data' 
nb_epoch = 150  # number of epoch at training stage
# nb_epoch_cont = 150  # number of epoch at training (cont) stage
batch_size = 16  # batch size
T = 24  # number of time intervals in one day
CACHEDATA = True  # cache data or NOT

lr = 0.00015  # learning rate
len_closeness = 3  # length of closeness dependent sequence
len_period = 1  # length of peroid dependent sequence
len_trend = 1  # length of trend dependent sequence

nb_flow = 2  # there are two types of flows: new-flow and end-flow
# divide data into two subsets: Train & Test, of which the test set is the
# last 10 days
days_test = 10
len_test = T*days_test
len_val = 2*len_test

map_height, map_width = 16, 8  # grid size
# For NYC Bike data, there are 81 available grid-based areas, each of
# which includes at least ONE bike station. Therefore, we modify the final
# RMSE by multiplying the following factor (i.e., factor).
nb_area = 81
m_factor = math.sqrt(1. * map_height * map_width / nb_area)
# print('factor: ', m_factor)

cache_folder = 'Autoencoder/model3' if model_name == 'model3' else 'Autoencoder'
path_cache = os.path.join(DATAPATH, 'CACHE', cache_folder)  # cache path
path_result = 'RET'
path_model = 'MODEL'
if os.path.isdir(path_result) is False:
    os.mkdir(path_result)
if os.path.isdir(path_model) is False:
    os.mkdir(path_model)
if CACHEDATA and os.path.isdir(path_cache) is False:
    os.mkdir(path_cache)

# load data
print("loading data...")
fname = os.path.join(path_cache, 'BikeNYC_C{}_P{}_T{}.h5'.format(
    len_closeness, len_period, len_trend))
if os.path.exists(fname) and CACHEDATA:
    X_train_all, Y_train_all, X_train, Y_train, \
    X_val, Y_val, X_test, Y_test, mmn, external_dim, \
    timestamp_train_all, timestamp_train, timestamp_val, timestamp_test = read_cache(
        fname, 'preprocessing_nyc.pkl')
    print("load %s successfully" % fname)
else:
    if (model_name == 'model3'):
        load_data = BikeNYC3d.load_data
    else:
        load_data = BikeNYC.load_data
    X_train_all, Y_train_all, X_train, Y_train, \
    X_val, Y_val, X_test, Y_test, mmn, external_dim, \
    timestamp_train_all, timestamp_train, timestamp_val, timestamp_test = load_data(
        T=T, nb_flow=nb_flow, len_closeness=len_closeness, len_period=len_period, len_trend=len_trend, len_test=len_test,
        len_val=len_val, preprocess_name='preprocessing_bikenyc.pkl', meta_data=True, datapath=DATAPATH)
    if CACHEDATA:
        cache(fname, X_train_all, Y_train_all, X_train, Y_train, X_val, Y_val, X_test, Y_test,
              external_dim, timestamp_train_all, timestamp_train, timestamp_val, timestamp_test)

print("\n days (test): ", [v[:8] for v in timestamp_test[0::T]])

def train_model(encoder_blocks, lstm_units, lr, batch_size, save_results=False, i=''):
    # get discrete parameters
    encoder_blocks = int(encoder_blocks)
    lstm_units = 2 ** int(lstm_units)
    batch_size = 16 * int(batch_size)

    filters = [32,64,16] if encoder_blocks==2 else [32,64,64,16]

    # build model
    m = models_dict[model_name]
    model = m.build_model(
        len_closeness, len_period, len_trend, nb_flow, map_height, map_width,
        external_dim=external_dim, lr=lr,
        encoder_blocks=2,
        filters=filters,
        lstm_units=lstm_units
        # save_model_pic=f'BikeNYC_{model_name}'
    )
    # model.summary()
    hyperparams_name = '{}.BikeNYC{}.c{}.p{}.t{}.encoderblocks_{}.lstm_{}.lr_{}.batchsize_{}'.format(
        model_name, i, len_closeness, len_period, len_trend, encoder_blocks, 
        lstm_units, lr, batch_size)
    fname_param = os.path.join('MODEL', '{}.best.h5'.format(hyperparams_name))

    early_stopping = EarlyStopping(monitor='val_rmse', patience=25, mode='min')
    model_checkpoint = ModelCheckpoint(
        fname_param, monitor='val_rmse', verbose=0, save_best_only=True, mode='min')

    # train model
    print("training model...")
    ts = time.time()
    if (i):
        np.random.seed(i*18)
        tf.random.set_seed(i*18)
    history = model.fit(X_train, Y_train,
                        epochs=nb_epoch,
                        batch_size=batch_size,
                        validation_data=(X_test,Y_test),
                        # callbacks=[early_stopping, model_checkpoint],
                        callbacks=[model_checkpoint],
                        verbose=0)
    model.save_weights(os.path.join(
        'MODEL', '{}.h5'.format(hyperparams_name)), overwrite=True)
    pickle.dump((history.history), open(os.path.join(
        path_result, '{}.history.pkl'.format(hyperparams_name)), 'wb'))
    print("\nelapsed time (training): %.3f seconds\n" % (time.time() - ts))

    # evaluate
    model.load_weights(fname_param)
    score = model.evaluate(
        X_test, Y_test, batch_size=Y_test.shape[0], verbose=0)
    print('Test score: %.6f rmse (norm): %.6f rmse (real): %.6f' %
            (score[0], score[1], score[1] * (mmn._max - mmn._min) / 2.))

    if (save_results):
        print('evaluating using the model that has the best loss on the valid set')
        model.load_weights(fname_param) # load best weights for current iteration
        
        Y_pred = model.predict(X_test) # compute predictions

        score = evaluate(Y_test, Y_pred, mmn, rmse_factor=1) # evaluate performance

        # save to csv
        csv_name = os.path.join('results',f'{model_name}_bikeNYC_results.csv')
        if not os.path.isfile(csv_name):
            if os.path.isdir('results') is False:
                os.mkdir('results')
            with open(csv_name, 'a', encoding = "utf-8") as file:
                file.write('iteration,'
                        'rsme_in,rsme_out,rsme_tot,'
                        'mape_in,mape_out,mape_tot,'
                        'ape_in,ape_out,ape_tot'
                        )
                file.write("\n")
                file.close()
        with open(csv_name, 'a', encoding = "utf-8") as file:
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
            pbounds={'encoder_blocks':(2, 3.999),
                     'lstm_units':(3, 5.999), #2**
                     'lr':(0.001, 0.0001),
                     'batch_size':(1, 2.999)}, #*16
                     verbose=2)
 
optimizer.maximize(init_points=10, n_iter=10)

# training-test-evaluation iterations with best params
targets = [e['target'] for e in optimizer.res]
best_index = targets.index(max(targets))
params = optimizer.res[best_index]['params']
for i in range(0,10):
    train_model(encoder_blocks=params['encoder_blocks'],
                lstm_units=params['lstm_units'],
                lr=params['lr'],
                batch_size=params['batch_size'],
                save_results=True,
                i=i)