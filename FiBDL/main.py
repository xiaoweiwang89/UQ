import tensorflow.compat.v1 as tf  # 使用1.0版本的方法
tf.disable_v2_behavior()  # 禁用2.0版本的方法

from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
# We fix the random seed
np.random.seed(1)
import pickle
import warnings
warnings.filterwarnings("ignore")
import math
from scipy.misc import logsumexp
from layers.utils import concat_func, add_func
import numpy as np
from layers.interaction import CIN, FGCNNLayer
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Sequential,Model
from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Conv2D, Flatten, Input,RepeatVector,Lambda
from tensorflow.python.keras.optimizers import SGD
from tensorflow.python.keras.regularizers import l2
from layers.core import DNN
import time


class net:

    def __init__(self, X_train, y_train, n_hidden, n_epochs=40,
                 normalize=False, X_test=None, y_test=None):

        """
            Constructor for the class implementing a Bayesian neural network
            trained with the probabilistic back propagation method.

            @param X_train      Matrix with the features for the training data.
            @param y_train      Vector with the target variables for the
                                training data.
            @param n_hidden     Vector with the number of neurons for each
                                hidden layer.
            @param n_epochs     Numer of epochs for which to train the
                                network. The recommended value 40 should be
                                enough.
            @param normalize    Whether to normalize the input features. This
                                is recommended unles the input vector is for
                                example formed by binary features (a
                                fingerprint). In that case we do not recommend
                                to normalize the features.
        """

        # We normalize the training data to have zero mean and unit standard
        # deviation in the training set if necessary

        if normalize:
            self.std_X_train = np.std(X_train, 0)
            self.std_X_train[self.std_X_train == 0] = 1
            self.mean_X_train = np.mean(X_train, 0)
        else:
            self.std_X_train = np.ones(X_train.shape[1])
            self.mean_X_train = np.zeros(X_train.shape[1])

        X_train = (X_train - np.full(X_train.shape, self.mean_X_train)) / \
                  np.full(X_train.shape, self.std_X_train)

        self.mean_y_train = np.mean(y_train)
        self.std_y_train = np.std(y_train)

        y_train_normalized = (y_train - self.mean_y_train) / self.std_y_train
        y_train_normalized = np.array(y_train_normalized, ndmin=2).T

        # We construct the network
        N = X_train.shape[0]
        dropout = 0.1
        batch_size = 256
        tau = 0.0871109017617  # obtained from BO
        lengthscale = 1e-2
        reg = lengthscale ** 2 * (1 - dropout) / (2. * N * tau)

        inputs = Input(shape=(X_train.shape[1],))
        drop1 = Dropout(0.05)(inputs)
        xx = RepeatVector(1)(drop1)
        xx_0 = FGCNNLayer(filters=(14, 16,), kernel_width=(7, 7,), new_maps=(3, 3,), pooling_width=(2, 2),)(xx)
        xx_00 = Flatten()(xx_0)

        cross_1 = CIN(layer_size=(128, 128), activation='relu', split_half=True, l2_reg=1e-5, seed=1024)(xx_0)
        cross_1 = Flatten()(cross_1)
        inter_1 = Dense(1, use_bias=False, activation=None)(cross_1)

        cross_2 = Dense(units=500, activation='relu')(xx_00)
        inter_2 = Dropout(0.05)(inputs=cross_2)
        inter_2 = Dense(units=500, activation='relu')(inter_2)
        inter_2 = Dropout(0.05)(inputs=inter_2)
        inter_2 = Dense(units=500, activation='relu')(inter_2)
        inter_2 = Dropout(0.05)(inputs=inter_2)
        inter_2 = Dense(1, use_bias=False, activation=None)(inter_2)

        inter = concat_func([inter_2, inter_1])

        output = Dense(1, activation='sigmoid', activity_regularizer=l2(reg))(inter)

        # input_list是list类型，model.fit中的X_train需要是dict类型
        model = Model(inputs, output)

        # model.compile(loss='mean_squared_error', optimizer='adam')
        model.compile('adam', 'binary_crossentropy',
                      metrics=['binary_crossentropy'], )
        model.summary()
        # We iterate the learning process
        start_time = time.time()

        #    X_train1 = {name: X_train[name] for name in feature_names}
        model.fit(X_train, y_train_normalized, batch_size=256, epochs=1, verbose=2)

        self.model = model
        self.tau = tau
        self.running_time = time.time() - start_time

    def predict(self, X_test, y_test):

        """  model.add(Dense(y_train_normalized.shape[1], W_regularizer=l2(reg)))                                             model.add(D
            Function for making predictions with the Bayesian neural network.

            @param X_test   The matrix of features for the test data


            @return m       The predictive mean for the test target variables.
            @return v       The predictive variance for the test target
                            variables.
            @return v_noise The estimated variance for the additive noise.

        """

        X_test = np.array(X_test, ndmin=2)
        y_test = np.array(y_test, ndmin=2).T

        # We normalize the test set

        X_test = (X_test - np.full(X_test.shape, self.mean_X_train)) / \
                 np.full(X_test.shape, self.std_X_train)

        # We compute the predictive mean and variance for the target variables
        # of the test data

        model = self.model

        # We compute the standard_pred rmse
        standard_pred = model.predict(X_test, batch_size=256, verbose=2)
        standard_pred = standard_pred * self.std_y_train + self.mean_y_train
        rmse_standard_pred = np.mean((y_test.squeeze() - standard_pred.squeeze()) ** 2.) ** 0.5

        # We compute the MC rmse
        T = 100
        predict_stochastic = K.function([model.layers[0].input, K.learning_phase()], [model.layers[-1].output])
        Yt_hat = np.array([predict_stochastic([X_test, 1]) for _ in range(T)])
        Yt_hat = Yt_hat * self.std_y_train + self.mean_y_train
        MC_pred = np.mean(Yt_hat, 0)
        rmse = np.mean((y_test.squeeze() - MC_pred.squeeze()) ** 2.) ** 0.5

        # We compute the MC mae
        mae_standard = np.mean(np.abs(y_test.squeeze() - standard_pred.squeeze()))

        MC_mae = np.mean(np.abs(y_test.squeeze() - MC_pred.squeeze()))

        # we compute the MC logloss
        logloss_standard = np.mean(
            -np.sum(y_test * np.log(standard_pred) + (1 - y_test) * np.log(1 - standard_pred)) / len(y_test))
        MC_logloss = np.mean(-np.sum(y_test * np.log(MC_pred) + (1 - y_test) * np.log(1 - MC_pred)) / len(y_test))

        print('ICME:')
        print('Standard rmse %f' % (rmse_standard_pred))
        print('MC rmse %f' % (rmse))

        print('Standard mae %f' % (mae_standard))
        print('MC_mae %f' % (MC_mae))

        print('Standard logloss %f' % (logloss_standard))
        print('MC logloss %f' % (MC_logloss))

        # We are done!
        return rmse_standard_pred, rmse,  mae_standard, MC_mae, logloss_standard, MC_logloss


if __name__ == "__main__":
    # We load the data

    data = pd.read_csv(
        '/Users/wangxiaowei/Documents/python/deepctr-0.7.3/deepctr/examples/ICME2019/input/ICME_data.csv')

    sparse_features = ['uid', 'user_city', 'item_id', 'author_id', 'item_city', 'channel', 'finish',
                       'music_id', 'did', ]
    dense_features = ['video_duration', 'creat_time']
    data[sparse_features] = data[sparse_features].fillna('-1', )
    data[dense_features] = data[dense_features].fillna(0,)

    target = ['like']

    feature = data.drop(columns=['like'])

    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])

    # We load the number of hidden units

    n_hidden = 30

    # We load the number of training epocs

    epochs = 50

    errors, MC_errors, std_maes, MC_maes, losses_std, MC_losses, times = [], [], [], [], [], [], []
    for i in range(10):
        # We load the indexes of the training and test sets

        X_train, X_test, y_train, y_test = train_test_split(feature, data['like'], test_size=0.2, random_state=2020)
        # We construct the network

        # We iterate the method

        network = net(X_train, y_train,
                      [n_hidden], normalize=True, n_epochs=10, X_test=X_test, y_test=y_test)
        running_time = network.running_time

        # We obtain the test RMSE and the test ll

    error, MC_error, std_mae, MC_mae, loss_std, MC_loss = network.predict(X_test, y_test)

    print(i)
    errors += [error]
    MC_errors += [MC_error]
    std_maes += [std_mae]
    MC_maes += [MC_mae]
    losses_std += [loss_std]
    MC_losses += [MC_loss]
    times += [running_time]

