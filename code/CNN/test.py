"""
Copyright 2017-2022 Department of Electrical and Computer Engineering
University of Houston, TX/USA
**********************************************************************************
Author:   Aryan Mobiny
Date:     7/1/2017
Comments: Run this file to test the best saved model
**********************************************************************************
"""

import tensorflow as tf
from utils import *
from AlexNet import Alexnet
import os
import h5py
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

save_dir = './checkpoints/'


def test(image_size=32,
         num_classes=2,
         num_channels=1):
    # load the test data
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    X_train, Y_train, X_valid, Y_valid = load_data(dim=2,
                                                   normalize='unity_based',
                                                   one_hot=True)
    y_true = np.argmax(Y_valid, axis=1)

    # load the model
    model = Alexnet(num_classes, image_size, num_channels)
    model.inference().accuracy_func().loss_func().train_func().pred_func()

    saver = tf.train.Saver()
    save_path = os.path.join(save_dir, 'model_86.91_71')

    with tf.Session() as sess:
        saver.restore(sess, save_path)
        print("Model restored.")
        y_pred_prob, acc_test, loss_test = validation(sess, model, X_valid, Y_valid, 10, compute_prob=True)
        print("Test loss: {0:.2f}, Test accuracy: {1:.01%}"
              .format(loss_test, acc_test))

    y_pred = np.argmax(y_pred_prob, 1)
    print(confusion_matrix(y_true, y_pred))
    Precision, Recall, thresholds = precision_recall_curve(y_true, y_pred_prob[:, 1])

    # Plot Precision-Recall curve
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    plt.plot(Recall, Precision, lw=2, color='navy', label='Precision-Recall curve')
    ax1.set_xlabel('Recall', size=18)
    ax1.set_ylabel('Precision', size=18)
    ax1.tick_params(labelsize=18)
    plt.ylim([0.5, 1.05])
    plt.xlim([0.0, 1.0])
    plt.show()

    # Compute precision and recall
    precision_recall(y_true, y_pred)

    # save the results
    h5f = h5py.File('alexnet_2d_results.h5', 'w')
    h5f.create_dataset('Precision', data=Precision)
    h5f.create_dataset('Recall', data=Recall)
    h5f.create_dataset('y_pred_prob', data=y_pred_prob)
    h5f.create_dataset('thresholds', data=thresholds)
    h5f.create_dataset('y_pred', data=y_pred)
    h5f.create_dataset('y_true', data=y_true)
    h5f.close()


if __name__ == '__main__':
    test(image_size=32,
         num_classes=2,
         num_channels=1)
