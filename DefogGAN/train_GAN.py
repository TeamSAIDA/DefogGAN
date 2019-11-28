import warnings
import keras
import keras.backend as K
import tensorflow as tf
from keras.utils.training_utils import multi_gpu_model
from keras.optimizers import Adam
from keras.utils import multi_gpu_model
from keras.models import Sequential
from keras.layers import Conv2D, Activation, LeakyReLU, BatchNormalization, Conv2DTranspose
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.engine.training import Model
from keras.engine.topology import Input
from tensorflow.python.client import device_lib
from sklearn.metrics import mean_squared_error
import numpy as np
import os
import pandas as pd
import math
import shutil
import csv
import random
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TKAgg')
import skimage.measure
warnings.filterwarnings('ignore')
#################################
"""
DefogGAN test Code
This source contains an 11 minute moment scene of 3500 replays to solve the defog problem.
It has a smaller amount than the data used in the paper.
Output MSE as a learning result. It also outputs images of fog, defog(epoch last model), defog(epoch best model) and ground truth.
"""
#################################
class DefogGAN():
    def __init__(self):
        self.project_dir = (os.path.dirname(__file__))
        self.data_path = os.path.join(self.project_dir, "./data/starCraft")
        self.result_work_path = os.path.join(self.project_dir, './result/DefogGAN')
        self.making_ing_path = os.path.join(self.project_dir, './result/DefogGAN')
        self.createFolder(self.result_work_path)
        is_multi_gpu = True
        self.is_validation_check = True
        self.save_weights = True
        #self.num_of_replay = 3500
        self.n_epochs = 1000
        self.batch_size = 512
        self.save_interval = 1000
        self.num_of_making_img = 5
        optimizer = Adam(0.0001, beta_1=0.5, beta_2=0.9)
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        self.generator = self.build_generator()
        fog_img = Input(shape=(82, 32, 32))
        gen_missing = self.generator(fog_img)
        self.discriminator.trainable = False
        valid = self.discriminator(gen_missing)
        self.combined = Model(fog_img, [gen_missing, valid])
        if is_multi_gpu:
            self.combined = multi_gpu_model(self.combined, gpus=self.get_count_of_gpu())
        weight = K.variable(np.array([0.75, 0.1875, 0.0468, 0.012, 0.003, 0.0007]))
        self.combined.compile(loss=[self.weighted_pyramidal_loss(weights=weight), 'binary_crossentropy'], loss_weights=[0.999, 0.001], optimizer=optimizer)
    def accumulate_resolution(self, x, threshhold=0.5):
        new = np.zeros((x.shape[0],32,32))
        new_enemy = np.zeros((x.shape[0],32,32))
        new_self = np.zeros((x.shape[0],32,32))
        for n in range(x.shape[0]):
            for w in range(x.shape[2]):
                for h in range(x.shape[3]):
                    is_enemy = False
                    is_self = False
                    for u in range(x.shape[1]):
                        if u <32:
                            # enemy
                            if not int(x[n][u][w][h] + threshhold)==0:
                                is_enemy = True
                                new_enemy[n][w][h] -= int(x[n][u][w][h] + threshhold)
                        else:
                            # self
                            if not int(x[n][u][w][h] + threshhold) == 0:
                                is_self = True
                                new_self[n][w][h] += int(x[n][u][w][h] + threshhold)
                    if is_enemy and is_self:
                        new[n][w][h] = 0
                    elif not is_enemy and not is_self:
                        new[n][w][h] = -30
                    elif is_enemy and not is_self:
                        new[n][w][h] = new_enemy[n][w][h]
                    elif not is_enemy and is_self:
                        new[n][w][h] = new_self[n][w][h]
        return new

    def make_pickle(self):
        for i in ['train', 'validation', 'test']:
            for j in ['x', 'y']:
                f = open('{}/{}_{}_data_set.csv'.format(self.data_path, j, i), 'r', encoding='utf-8')
                read_csv_file = csv.reader(f)
                is_first = True
                for line in read_csv_file:
                    if is_first:
                        tensor = np.zeros((int(line[0]), int(line[1]), int(line[2]), int(line[3])))
                        is_first = False
                    else:
                        tensor[int(line[0])][int(line[1])][int(line[2])][int(line[3])] = float(line[4])
                        #print(int(line[0]), int(line[1]), int(line[2]), int(line[3]), tensor[int(line[0])][int(line[1])][int(line[2])][int(line[3])])
                f.close()
                with open('{}/{}_{}_dataset.pkl'.format(self.data_path, j, i), 'wb') as f:
                    pickle.dump(tensor, f, pickle.HIGHEST_PROTOCOL)

    def get_pickle_data(self):
        with open('{}/x_train_dataset.pkl'.format(self.data_path), 'rb') as f:
            x_train = pickle.load(f)
        with open('{}/x_validation_dataset.pkl'.format(self.data_path), 'rb') as f:
            x_validation = pickle.load(f)
        with open('{}/x_test_dataset.pkl'.format(self.data_path), 'rb') as f:
            x_test = pickle.load(f)

        with open('{}/y_train_dataset.pkl'.format(self.data_path), 'rb') as f:
            y_train = pickle.load(f)
        with open('{}/y_validation_dataset.pkl'.format(self.data_path), 'rb') as f:
            y_validation = pickle.load(f)
        with open('{}/y_test_dataset.pkl'.format(self.data_path), 'rb') as f:
            y_test = pickle.load(f)
        return x_train, x_validation, x_test, y_train, y_validation, y_test

    def get_sample_data(self):
        f = open('{}/sample_data.csv'.format(self.data_path), 'r', encoding='utf-8')
        read_csv_file = csv.reader(f)
        tensor_fog = np.zeros((self.num_of_replay,82,32,32))
        tensor_real= np.zeros((self.num_of_replay,66,32,32))
        for line in read_csv_file:
            if len(line) == 1:
                num_of_replay = int(line[0])
            if len(line) == 5:
                tensor_fog[num_of_replay][int(line[0])][int(line[1])][int(line[2])] = int(line[3])
                if int(line[0]) < 66:
                    tensor_real[num_of_replay][int(line[0])][int(line[1])][int(line[2])] = int(line[4])
        f.close()
        # shuffle tensor index
        train_set_index = random.sample(range(0, self.num_of_replay), int(self.num_of_replay * self.train_rate))
        temp_set = [i for i in range(0, self.num_of_replay) if not i in train_set_index]
        validation_set_index = temp_set[:int(self.num_of_replay * self.validation_rate)]
        test_set_index = temp_set[int(self.num_of_replay * self.test_rate):]

        return tensor_fog[train_set_index], tensor_fog[validation_set_index], tensor_fog[test_set_index], tensor_real[train_set_index], tensor_real[validation_set_index], tensor_real[test_set_index]
    def check_pickle(self):
        for i in ['train', 'validation', 'test']:
            for j in ['x', 'y']:
                fname = '{}/{}_{}_dataset.pkl'.format(self.data_path, j, i)
                if not os.path.isfile(fname):
                    return False
        return True

    def train_defogGAN(self):
        min_loss = 99999999999
        valid = np.ones((self.batch_size, 1))
        fake = np.zeros((self.batch_size, 1))
        #x_train, x_validation, x_test, y_train, y_validation, y_test = self.get_sample_data()
        is_pickle = self.check_pickle()
        if not is_pickle:
            self.make_pickle()
        x_train, x_validation, x_test, y_train, y_validation, y_test = self.get_pickle_data()
        best_img = None
        last_epoch_img = None
        for epoch in range(self.n_epochs + 1):
            n_batches = int(x_train.shape[0] / self.batch_size)
            for i in range(math.ceil(n_batches)):
                start_batch = i * self.batch_size
                end_batch = (1 + i) * self.batch_size
                end_batch = x_train.shape[0] if end_batch > x_train.shape[0] else (1 + i) * self.batch_size
                images_train_x = x_train[start_batch: end_batch, :, :, :]
                images_train_y = y_train[start_batch: end_batch, :, :, :]
                gen_missing = self.generator.predict(images_train_x)
                d_loss_real = self.discriminator.train_on_batch(images_train_y, valid)
                d_loss_fake = self.discriminator.train_on_batch(gen_missing, fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                #  Train Generator
                g_loss = self.combined.train_on_batch(images_train_x, [images_train_y, valid])
            if self.is_validation_check:
                validation_recon_img = self.generator.predict(x_validation)
                validation_loss = self.get_MSE_Value(validation_recon_img, y_validation)
            else:
                validation_loss = 0
            print("%d [D loss: %f, acc: %.2f%%] [G loss: %f, mse: %f, validation_mse : %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss[0], g_loss[1], validation_loss))
            if epoch % self.save_interval == 0:
                last_epoch_img = self.generator.predict(x_test)
                if self.save_weights:
                    self.createFolder('{}/interval_model/'.format(self.result_work_path))
                    self.generator.save_weights('{}/interval_model/model_weight_{}.h5'.format(self.result_work_path, epoch))

            if min_loss > validation_loss and self.is_validation_check:
                min_loss = validation_loss
                best_img = self.generator.predict(x_test)
                if self.save_weights:
                    if os.path.exists('{}/best_model'.format(self.result_work_path)):
                        shutil.rmtree('{}/best_model'.format(self.result_work_path))
                    self.createFolder('{}/best_model/'.format(self.result_work_path))
                    self.generator.save_weights('{}/best_model/model_weight_{}.h5'.format(self.result_work_path, epoch))
        self.make_test(x_test, last_epoch_img, best_img, y_test)
    def make_test(self, x_test, last_epoch_img, best_img, y_test):
        n = self.num_of_making_img
        x_test = x_test[:n]
        x_test = x_test[:,:66,:,:]
        last_epoch_img = last_epoch_img[:n]
        best_img = best_img[:n]
        y_test = y_test[:n]
        result = np.zeros((0, n, 32, 32))  # model , index of replay , x, y
        result = np.concatenate((result, self.accumulate_resolution(x_test).reshape(1, n, 32, 32)), axis=0)
        result = np.concatenate((result, self.accumulate_resolution(last_epoch_img).reshape(1, n, 32, 32)), axis=0)
        result = np.concatenate((result, self.accumulate_resolution(best_img).reshape(1, n, 32, 32)), axis=0)
        result = np.concatenate((result, self.accumulate_resolution(y_test).reshape(1, n, 32, 32)), axis=0)

        print(result.shape)
        self.make_img(self.making_ing_path, result)
        print('Success make image')
    def make_img(self, path, map):
        print(map.shape)  # (7,30,32,32)
        plt_threshhold = -20
        model_names = ['fog_exposed', 'last_epoch', 'best_epoch', 'Ground_truth']
        fig, axn = plt.subplots(map.shape[1], map.shape[0], sharex=True, sharey=True,
                                figsize=(map.shape[0] * 1.3, map.shape[1] * 1.3))
        cbar_ax = fig.add_axes([.91, .3, .03, .4])
        fig.suptitle('GAN\'s compare[enemy(red): positive num, allies(green): negative num, both(yellow): 0]', fontsize=16)
        for i, ax in enumerate(axn.flat):

            model_index = i % map.shape[0]
            unit_index = int(i / map.shape[0])
            if i < map.shape[0]:
                ax.set_title(model_names[model_index], fontsize=10)
            matrix = map[model_index][unit_index]
            if plt_threshhold == 0:
                sns.heatmap(matrix, ax=ax, cbar=i == 0, annot=True, fmt='.1f', cmap=plt.cm.YlGnBu,
                            cbar_ax=None if i else cbar_ax)
            else:
                cbar_kws = {'ticks': [-3, -2, -1, 0, 1, 2, 3], 'drawedges': True}
                sns.heatmap(matrix, mask=(matrix < plt_threshhold), ax=ax, cbar=i == 0, annot=False, square=True,
                            fmt='.1f', xticklabels=False, yticklabels=False, vmin=-3.5, vmax=3.5, cbar_kws=cbar_kws,
                            cmap=plt.get_cmap('RdYlGn', 7), cbar_ax=None if i else cbar_ax)
        count = 0
        for ax in axn.flat:
            model_index = count % map.shape[0]
            unit_index = int(count / map.shape[0])
            if model_index == 0:
                ax.set(ylabel='replay {}'.format(unit_index))
            ax.axhline(y=0, color='k', linewidth=1)
            ax.axhline(y=32, color='k', linewidth=2)
            ax.axvline(x=0, color='k', linewidth=1)
            ax.axvline(x=32, color='k', linewidth=2)
            count += 1
        plt.subplots_adjust(hspace=0.03, wspace=0.03)
        plt.savefig(path)
        plt.close('all')
    def build_generator(self):
        input_channel = 82
        output_channel = 66
        input_shape = (input_channel, 32, 32)
        img_input = Input(shape=input_shape, name='input')
        depth = input_channel
        # In: 100
        # Out: dim x dim x depth
        c1 = Conv2D(depth * 2, (3, 3), strides=(1, 1), activation='relu', input_shape=input_shape, padding='same', data_format='channels_first')(img_input)
        b1 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                beta_initializer='zeros',
                                gamma_initializer='ones', moving_mean_initializer='zeros',
                                moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
                                beta_constraint=None, gamma_constraint=None)(c1)
        act1 = Activation('relu')(b1)
        c2 = Conv2D(depth * 2, (3, 3), strides=(2, 2), activation='relu', padding='same', data_format='channels_first')(act1)
        b2 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                beta_initializer='zeros',
                                gamma_initializer='ones', moving_mean_initializer='zeros',
                                moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
                                beta_constraint=None, gamma_constraint=None)(c2)
        act2 = Activation('relu')(b2)
        c3 = Conv2D(depth * 4, (3, 3), strides=(1, 1), activation='relu', padding='same', data_format='channels_first')(act2)
        b3 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                beta_initializer='zeros',
                                gamma_initializer='ones', moving_mean_initializer='zeros',
                                moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
                                beta_constraint=None, gamma_constraint=None)(c3)
        act3 = Activation('relu')(b3)
        c4 = Conv2D(depth * 4, (3, 3), strides=(2, 2), activation='relu', padding='same', data_format='channels_first')(act3)
        b4 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                beta_initializer='zeros',
                                gamma_initializer='ones', moving_mean_initializer='zeros',
                                moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
                                beta_constraint=None, gamma_constraint=None)(c4)
        act4 = Activation('relu')(b4)
        c5 = Conv2D(depth * 8, (3, 3), strides=(1, 1), activation='relu', padding='same', data_format='channels_first')(act4)
        b5 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                beta_initializer='zeros',
                                gamma_initializer='ones', moving_mean_initializer='zeros',
                                moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
                                beta_constraint=None, gamma_constraint=None)(c5)
        act5 = Activation('relu')(b5)
        c6 = Conv2D(depth * 8, (3, 3), strides=(1, 1), activation='relu', padding='same', data_format='channels_first')(act5)
        b6 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                beta_initializer='zeros',
                                gamma_initializer='ones', moving_mean_initializer='zeros',
                                moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
                                beta_constraint=None, gamma_constraint=None)(c6)
        act6 = Activation('relu')(b6)
        c7 = Conv2D(depth * 8, (3, 3), strides=(2, 2), activation='relu', padding='same', data_format='channels_first')(act6)
        b7 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                beta_initializer='zeros',
                                gamma_initializer='ones', moving_mean_initializer='zeros',
                                moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
                                beta_constraint=None, gamma_constraint=None)(c7)
        act7 = Activation('relu')(b7)
        ct1 = Conv2DTranspose(depth * 8, (3, 3), strides=(2, 2), activation='relu', padding='same', data_format='channels_first')(act7)
        act8 = Activation('relu')(ct1)
        act8_output = Lambda(lambda x: x, name='act8_output')(act8)
        act8_output = keras.layers.Add()([act6, act8_output])
        ct2 = Conv2DTranspose(depth * 8, (3, 3), strides=(1, 1), activation='relu', padding='same',data_format='channels_first')(act8_output)
        act9 = Activation('relu')(ct2)
        act9_output = Lambda(lambda x: x, name='act9_output')(act9)
        act9_output = keras.layers.Add()([act5, act9_output])
        ct3 = Conv2DTranspose(depth * 4, (3, 3), strides=(1, 1), activation='relu', padding='same', data_format='channels_first')(act9_output)
        act10 = Activation('relu')(ct3)
        act10_output = Lambda(lambda x: x, name='act10_output')(act10)
        act10_output = keras.layers.Add()([act4, act10_output])
        ct4 = Conv2DTranspose(depth * 4, (3, 3), strides=(2, 2), activation='relu', padding='same', data_format='channels_first')(act10_output)
        act11 = Activation('relu')(ct4)
        act11_output = Lambda(lambda x: x, name='act11_output')(act11)
        act11_output = keras.layers.Add()([act3, act11_output])
        ct5 = Conv2DTranspose(depth * 2, (3, 3), strides=(1, 1), activation='relu', padding='same', data_format='channels_first')(act11_output)
        act12 = Activation('relu')(ct5)
        act12_output = Lambda(lambda x: x, name='act12_output')(act12)
        act12_output = keras.layers.Add()([act2, act12_output])
        ct6 = Conv2DTranspose(depth * 2, (3, 3), strides=(2, 2), activation='relu', padding='same', data_format='channels_first')(act12_output)
        act13 = Activation('relu')(ct6)
        act13_output = Lambda(lambda x: x, name='act13_output')(act13)
        act13_output = keras.layers.Add()([act1, act13_output])
        ct7 = Conv2DTranspose(depth, (3, 3), strides=(1, 1), activation='relu', padding='same', data_format='channels_first')(act13_output)
        act14 = Activation('relu')(ct7)
        act14_output = Lambda(lambda x: x, name='output')(act14)
        act14_output = keras.layers.Add()([img_input, act14_output])
        ct8 = Conv2DTranspose(output_channel, (3, 3), strides=(1, 1), activation='relu', padding='same', data_format='channels_first')(act14_output)
        act15 = Activation('relu')(ct8)
        img_output = act15
        model = Model(inputs=[img_input], outputs=[img_output])
        model.summary()
        return model
    def build_discriminator(self):
        D = Sequential()
        discriminator_input_channel = 66
        depth = discriminator_input_channel
        dropout = 0.4
        input_shape = (discriminator_input_channel, 32, 32)
        D.add(Conv2D(depth * 1, (3, 3), strides=(2, 2), input_shape=input_shape, padding='same', data_format='channels_first'))
        D.add(LeakyReLU(alpha=0.2))
        D.add(Dropout(dropout))
        D.add(Conv2D(depth * 2, (3, 3), strides=(2, 2), padding='same', data_format='channels_first'))
        D.add(LeakyReLU(alpha=0.2))
        D.add(Dropout(dropout))
        D.add(Conv2D(depth * 4, (3, 3), strides=(2, 2), padding='same', data_format='channels_first'))
        D.add(LeakyReLU(alpha=0.2))
        D.add(Dropout(dropout))
        D.add(Flatten())
        D.add(Dense(1))
        D.add(Activation('sigmoid'))
        D.summary()
        return D

    def weighted_pyramidal_loss(self, weights):
        def pyramidal_loss(y_true, y_pred):
            yt_2 = keras.layers.AveragePooling2D((2, 2))(y_true) * 2
            yt_4 = keras.layers.AveragePooling2D((4, 4))(y_true) * 4
            yt_8 = keras.layers.AveragePooling2D((8, 8))(y_true) * 8
            yt_16 = keras.layers.AveragePooling2D((16, 16))(y_true) * 16
            yt_32 = keras.layers.AveragePooling2D((32, 32))(y_true) * 32

            yp_2 = keras.layers.AveragePooling2D((2, 2))(y_pred) * 2
            yp_4 = keras.layers.AveragePooling2D((4, 4))(y_pred) * 4
            yp_8 = keras.layers.AveragePooling2D((8, 8))(y_pred) * 8
            yp_16 = keras.layers.AveragePooling2D((16, 16))(y_pred) * 16
            yp_32 = keras.layers.AveragePooling2D((32, 32))(y_pred) * 32

            loss_0 = keras.losses.mean_squared_error(y_true, y_pred)
            loss_2 = keras.losses.mean_squared_error(yt_2, yp_2)
            loss_4 = keras.losses.mean_squared_error(yt_4, yp_4)
            loss_8 = keras.losses.mean_squared_error(yt_8, yp_8)
            loss_16 = keras.losses.mean_squared_error(yt_16, yp_16)
            loss_32 = keras.losses.mean_squared_error(yt_32, yp_32)
            loss_0 = tf.reduce_mean(loss_0, axis=[1, 2])
            loss_2 = tf.reduce_mean(loss_2, axis=[1, 2])
            loss_4 = tf.reduce_mean(loss_4, axis=[1, 2])
            loss_8 = tf.reduce_mean(loss_8, axis=[1, 2])
            loss_16 = tf.reduce_mean(loss_16, axis=[1, 2])
            loss_32 = tf.reduce_mean(loss_32, axis=[1, 2])
            loss = weights[0] * loss_0 + \
                   weights[1] * loss_2 + \
                   weights[2] * loss_4 + \
                   weights[3] * loss_8 + \
                   weights[4] * loss_16 + \
                   weights[5] * loss_32
            return loss

        return pyramidal_loss
    def get_count_of_gpu(self):
        device_list = device_lib.list_local_devices()
        gpu_count = 0
        for d in device_list:
            if d.device_type == 'GPU':
                gpu_count += 1
        return int(gpu_count)

    def createFolder(self, directory):
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except OSError:
            print('Error: Creating directory. ' + directory)
    def get_MSE_Value(self, x, y):
        MSE_GAN = 0
        MSE_DIV = x.shape[0] * x.shape[1]
        for n in range(x.shape[0]):
            for c in range(x.shape[1]):
                MSE_GAN += mean_squared_error(x[n][c], y[n][c])
        MSE = MSE_GAN / MSE_DIV
        return MSE
def main():
    defogGAN = DefogGAN()
    defogGAN.train_defogGAN()


if __name__ == '__main__':
    main()