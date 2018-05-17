import cv2
import h5py
import os
import numpy as np
from config import Config

# conditions = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
#               'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']


class chest_xray():

    def __init__(self, config):
        self.config = config
        conditions = [5]
        self.load_data(config.original_size, config.num_channels, conditions, balance=config.balance)

    def load_data(self, image_size, num_channels, conditions, balance=True):
        dir_path_parent = os.path.dirname(os.getcwd())
        dir_path_train = dir_path_parent + '/data/chest256_train_801010_no_normal.h5'
        dir_path_valid = dir_path_parent + '/data/chest256_val_801010_no_normal.h5'
        dir_path_test = dir_path_parent + '/data/chest256_test_801010_no_normal.h5'
        h5f_train = h5py.File(dir_path_train, 'r')
        x_train = h5f_train['X_train'][:]
        y_train = h5f_train['Y_train'][:, conditions]
        h5f_train.close()
        m = np.mean(x_train, axis=0)
        s = np.std(x_train, axis=0)

        if balance:
            y, y_idx = np.sort(y_train), np.argsort(y_train)
            x = x_train[y_idx]
            pos_num = int(np.sum(y))
            x_train = np.concatenate((x[:pos_num+582], x[-pos_num:]), axis=0)
            self.y_train = np.concatenate((y[:pos_num+582], y[-pos_num:]), axis=0)
        else:
            self.y_train = y_train

        h5f_valid = h5py.File(dir_path_valid, 'r')
        x_valid = h5f_valid['X_val'][:]
        self.y_valid = h5f_valid['Y_val'][:, conditions]
        h5f_valid.close()

        x_train = (x_train - m) / s
        x_valid = (x_valid - m) / s

        h5f_test = h5py.File(dir_path_test, 'r')
        x_test = h5f_test['X_test'][:]
        self.y_test = h5f_test['Y_test'][:, conditions]
        h5f_test.close()
        x_test = (x_test - m) / s

        x_train = resize_image(x_train, size=128)
        x_valid = resize_image(x_valid, size=128)
        x_test = resize_image(x_test, size=128)
        image_size = 128

        self.x_test = x_test.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
        self.x_train = x_train.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
        self.x_valid = x_valid.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)

        return self


def resize_image(x, size=128):
    x_r = np.zeros((x.shape[0], size, size))
    for i in range(x.shape[0]):
        feat = x[i, :, :]
        x_r[i, :, :] = cv2.resize(feat, dsize=(size, size), interpolation=cv2.INTER_CUBIC)
    return x_r

def get_next_batch(x, y, start, end):
    x_batch = x[start:end]
    y_batch = y[start:end]
    return x_batch, y_batch


def randomize(x, y):
    """ Randomizes the order of data samples and their corresponding labels"""
    permutation = np.random.permutation(y.shape[0])
    shuffled_x = x[permutation, :, :, :]
    shuffled_y = y[permutation]
    return shuffled_x, shuffled_y
