"""
Copyright 2017-2022 Department of Electrical and Computer Engineering
University of Houston, TX/USA
**********************************************************************************
Author:   Aryan Mobiny
Date:     6/1/2017
Comments: Python utility functions
**********************************************************************************
"""

import numpy as np
import random
import scipy.ndimage
import h5py
import os

def normal(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def load_data(dim=2, normalize='unity_based', one_hot=True):
    if dim == 2:
        h5f = h5py.File('/home/exx/Desktop/Lung_Nodule_data/Lung_Nodule_2d.h5', 'r')
        X_train = h5f['X_train'][:]
        Y_train = h5f['Y_train'][:]
        X_valid = h5f['X_valid'][:]
        Y_valid = h5f['Y_valid'][:]
        h5f.close()
    elif dim == 3:
        h5f = h5py.File('/home/exx/Desktop/Lung_Nodule_data/Lung_Nodule.h5', 'r')
        X_train = h5f['X_train'][:]
        Y_train = h5f['Y_train'][:]
        X_valid = h5f['X_valid'][:]
        Y_valid = h5f['Y_valid'][:]

    image_size, num_classes, num_channels = X_train.shape[1], len(np.unique(Y_train)), 1
    X_train = np.maximum(np.minimum(X_train, 4096.), 0.)
    X_valid = np.maximum(np.minimum(X_valid, 4096.), 0.)

    if normalize == 'standard':
        m = np.mean(X_train)
        s = np.std(X_train)
        X_train = (X_train - m) / s
        X_valid = (X_valid - m) / s
    elif normalize == 'unity_based':
        X_train = np.asanyarray([normal(X_train[i]) for i in range(len(X_train))])
        X_valid = np.asanyarray([normal(X_valid[i]) for i in range(len(X_valid))])

    if one_hot:
        X_train, Y_train = reformat(X_train, Y_train, image_size, num_channels, num_classes)
        X_valid, Y_valid = reformat(X_valid, Y_valid, image_size, num_channels, num_classes)
    elif not one_hot:
        X_train, _ = reformat(X_train, Y_train, image_size, num_channels, num_classes)
        X_valid, _ = reformat(X_valid, Y_valid, image_size, num_channels, num_classes)

    return X_train, Y_train, X_valid, Y_valid


def randomize(x, y):
    """ Randomizes the order of data samples and their corresponding labels"""
    permutation = np.random.permutation(y.shape[0])
    shuffled_x = x[permutation, :, :, :]
    shuffled_y = y[permutation]
    return shuffled_x, shuffled_y


def reformat(x, y, img_size, num_ch, num_class):
    """ Reformats the data to the format acceptable for the conv layers"""
    dataset = x.reshape(
        (-1, img_size, img_size, num_ch)).astype(np.float32)
    labels = (np.arange(num_class) == y[:, None]).astype(np.float32)
    return dataset, labels


def precision_recall(y_true, y_pred):
    """
    Computes the precision and recall values for the positive class
    :param y_true: true labels
    :param y_pred: predicted labels
    """
    TP = FP = FN = TN = 0
    for i in range(len(y_pred)):
        if y_true[i] == 1 and y_pred[i] == 1:
            TP += 1
        elif y_true[i] == 0 and y_pred[i] == 1:
            FP += 1
        elif y_true[i] == 1 and y_pred[i] == 0:
            FN += 1
        elif y_true[i] == 0 and y_pred[i] == 0:
            TN += 1
    precision = (TP * 100.0) / (TP + FP)
    recall = (TP * 100.0) / (TP + FN)
    print('Precision: {0:.2f}'.format(precision))
    print('Recall: {0:.2f}'.format(recall))


def get_next_batch(x, y, start, end):
    x_batch = x[start:end]
    y_batch = y[start:end]
    return x_batch, y_batch


def save_to():
    """
    Creating the handles for saving the results in a .csv file
    :return:
    """
    if not os.path.exists(args.results):
        os.mkdir(args.results)
    if not os.path.exists(args.results + args.dataset):
        os.mkdir(args.results + args.dataset)
    if not os.path.exists(args.results + args.dataset + args.run):
        os.mkdir(args.results + args.dataset + args.run)
    if args.mode == 'train':
        train_path = args.results + args.dataset + args.run + '/' + 'train.csv'
        val_path = args.results + args.dataset + args.run + '/' + 'validation.csv'
        pr_path = args.results + args.dataset + args.run + '/' + 'pr.csv'

        if os.path.exists(train_path):
            os.remove(train_path)
        if os.path.exists(val_path):
            os.remove(val_path)
        if os.path.exists(pr_path):
            os.remove(pr_path)

        f_train = open(train_path, 'w')
        f_train.write('step,accuracy,loss\n')
        f_val = open(val_path, 'w')
        f_val.write('epoch,accuracy,loss\n')
        f_pr = open(pr_path, 'w')
        f_pr.write('epoch,precision,recall\n')
        return f_train, f_val, f_pr
    else:
        test_path = args.results + args.dataset + args.run + '/test.csv'
        if os.path.exists(test_path):
            os.remove(test_path)
        f_test = open(test_path, 'w')
        f_test.write('accuracy,loss\n')
        return f_test


def random_rotation_2d(batch, max_angle):
    """ Randomly rotate an image by a random angle (-max_angle, max_angle).

    Arguments:
    max_angle: `float`. The maximum rotation angle.

    Returns:
    batch of rotated 2D images
    """
    size = batch.shape
    batch = np.squeeze(batch)
    batch_rot = np.zeros(batch.shape)
    for i in range(batch.shape[0]):
        if bool(random.getrandbits(1)):
            image = np.squeeze(batch[i])
            angle = random.uniform(-max_angle, max_angle)
            batch_rot[i] = scipy.ndimage.interpolation.rotate(image, angle, mode='nearest', reshape=False)
        else:
            batch_rot[i] = batch[i]
    return batch_rot.reshape(size)


def validation(sess, model, x, y, split_num, compute_prob=False):
    all_acc = all_loss = np.array([])
    x_split, y_split = np.split(x, split_num), np.split(y, split_num)
    if compute_prob:
        y_pred_prob = np.zeros((0, 2))
        for i in range(split_num):
            feed_dict = {model.x: x_split[i], model.y: y_split[i], model.keep_prob: 1}
            y, acc_, loss_ = sess.run([model.probs, model.accuracy, model.loss], feed_dict=feed_dict)
            all_acc = np.append(all_acc, acc_)
            all_loss = np.append(all_loss, loss_)
            y_pred_prob = np.concatenate((y_pred_prob, y), axis=0)
        return y_pred_prob, np.mean(all_acc), np.mean(all_loss)

    else:
        for i in range(split_num):
            feed_dict = {model.x: x_split[i], model.y: y_split[i], model.keep_prob: 1}
            acc_, loss_ = sess.run([model.accuracy, model.loss], feed_dict=feed_dict)
            all_acc = np.append(all_acc, acc_)
            all_loss = np.append(all_loss, loss_)
        return np.mean(all_acc), np.mean(all_loss)


def add_noise(batch, mean=0, var=0.1, amount=0.01, mode='pepper'):
    original_size = batch.shape
    batch = np.squeeze(batch)
    batch_noisy = np.zeros(batch.shape)
    for ii in range(batch.shape[0]):
        image = np.squeeze(batch[ii])
        if mode == 'gaussian':
            gauss = np.random.normal(mean, var, image.shape)
            image = image + gauss
        elif mode == 'pepper':
            num_pepper = np.ceil(amount * image.size)
            coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
            image[coords] = 0
        elif mode == "s&p":
            s_vs_p = 0.5
            # Salt mode
            num_salt = np.ceil(amount * image.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
            image[coords] = 1
            # Pepper mode
            num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
            image[coords] = 0
        batch_noisy[ii] = image
    return batch_noisy.reshape(original_size)
