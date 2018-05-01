"""
Copyright 2017-2022 Department of Electrical and Computer Engineering
University of Houston, TX/USA
**********************************************************************************
Author:   Aryan Mobiny
Date:     6/1/2017
Comments: Run this file to train the Alexnet model and save the best trained model
**********************************************************************************
"""

from datetime import datetime
import time
import tensorflow as tf
from utils import *
from AlexNet import Alexnet
import os

now = datetime.now()
logs_path = "./graph/" + now.strftime("%Y%m%d-%H%M%S")
save_dir = './checkpoints/'


def train(image_size=32,
          num_classes=2,
          num_channels=1,
          num_epochs=200,
          batch_size=16,
          display=100):

    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # load the test data
    X_train, Y_train, X_valid, Y_valid = load_data(dim=2,
                                                   normalize='unity_based',
                                                   one_hot=True)
    # Creating the alexnet model
    model = Alexnet(num_classes, image_size, num_channels)
    model.inference().accuracy_func().loss_func().train_func()

    # Saving the best trained model (based on the validation accuracy)
    saver = tf.train.Saver()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, 'model_')
    best_validation_accuracy = 0

    acc_batch_all = loss_batch_all = np.array([])
    sum_count = 0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print("Initialized")
        merged = tf.summary.merge_all()
        batch_writer = tf.summary.FileWriter(logs_path + '/batch/', sess.graph)
        valid_writer = tf.summary.FileWriter(logs_path + '/valid/')
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            print('-----------------------------------------------------------------------------')
            print('Epoch: {}'.format(epoch + 1))
            X_train, Y_train = randomize(X_train, Y_train)
            step_count = int(len(X_train) / batch_size)
            for step in range(step_count):
                start = step * batch_size
                end = (step + 1) * batch_size
                X_batch, Y_batch = get_next_batch(X_train, Y_train, start, end)
                X_batch = random_rotation_2d(X_batch, 90.0)
                X_batch = add_noise(X_batch, mode='pepper', amount=0.05)
                model.is_train = False
                feed_dict_batch = {model.x: X_batch, model.y: Y_batch, model.keep_prob: 0.5}

                _, acc_batch, loss_batch = sess.run([model.train_op, model.accuracy, model.loss],
                                                    feed_dict=feed_dict_batch)
                acc_batch_all = np.append(acc_batch_all, acc_batch)
                loss_batch_all = np.append(loss_batch_all, loss_batch)

                if step > 0 and not (step % display):
                    mean_acc = np.mean(acc_batch_all)
                    mean_loss = np.mean(loss_batch_all)
                    print(
                        "Step {0}, training loss: {1:.5f}, training accuracy: {2:.01%}".format(step, mean_loss,
                                                                                               mean_acc))
                    summary_tr = tf.Summary(value=[tf.Summary.Value(tag='Accuracy', simple_value=mean_acc)])
                    batch_writer.add_summary(summary_tr, sum_count * display)
                    summary_tr = tf.Summary(value=[tf.Summary.Value(tag='Loss', simple_value=mean_loss)])
                    batch_writer.add_summary(summary_tr, sum_count * display)
                    summary = sess.run(merged, feed_dict=feed_dict_batch)
                    batch_writer.add_summary(summary, sum_count * display)
                    sum_count += 1
                    acc_batch_all = loss_batch_all = np.array([])

            model.is_train = False
            acc_valid, loss_valid = validation(sess, model, X_valid, Y_valid, 10, compute_prob=False)
            summary_valid = tf.Summary(value=[tf.Summary.Value(tag='Accuracy', simple_value=acc_valid)])
            valid_writer.add_summary(summary_valid, sum_count * display)
            if acc_valid > best_validation_accuracy:
                # Update the best-known validation accuracy.
                best_validation_accuracy = acc_valid
                best_epoch = epoch
                # Save all variables of the TensorFlow graph to file.
                saver.save(sess=sess, save_path=save_path + str("%.2f" % (acc_valid*100)) + '_' + str(best_epoch))
                # A string to be printed below, shows improvement found.
                improved_str = '*'
            else:
                # An empty string to be printed below.
                # Shows that no improvement was found.
                improved_str = ''
            epoch_time = time.time() - epoch_start_time
            print("Epoch {0}, run time: {1:.1f} seconds, validation loss: {2:.2f}, validation accuracy: {3:.01%}{4}"
                  .format(epoch + 1, epoch_time, loss_valid, acc_valid, improved_str))


if __name__ == '__main__':
    train(image_size=32,
          num_classes=2,
          num_channels=1,
          num_epochs=200,
          batch_size=16,
          display=100)
