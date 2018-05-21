""" Train model.

--load          checkpoint directory if you want to continue training a model
--task          task to train on {'mnist', 'translated' or 'cluttered'}
--num_glimpses  # glimpses (fixed)
--n_patches     # resolutions extracted for each glimpse

"""

import tensorflow as tf
import numpy as np
import argparse
from datetime import datetime
import pickle
from RAM import RAM
# from DRAM import DRAM
# from DRAM_loc import DRAMl
from config import Config

from tensorflow.examples.tutorials.mnist import input_data
from read_chestxray import chest_xray
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"

if __name__ == '__main__':
    # a='./experiments/task=org256x256_model=ram_conv=True_n_glimpses=50_fovea=12x12_std=0.05_123330_context=True_lr=0.0005-1e-05_p_labels=1_5/'
    a= None
    # ----- parse command line -----
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', '-t', type=str, default='org',
                        help='Task - ["org","translated","cluttered", "cluttered_var"].')

    parser.add_argument('--weighted_loss', default=True,
                        help='Decide to use weighted loss or not')
    parser.add_argument('--balance', default=False,
                        help='balance the positive and negative classes')

    parser.add_argument('--model', '-m', type=str, default='ram',
                        help='Model - "RAM" or "DRAM".')
    parser.add_argument('--load', '-l', type=str, default=a,
                        help='Load model form directory.')
    parser.add_argument('--use_context', default=True, action='store_true',
                        help='Use context network (True) or not (False)')
    parser.add_argument('--convnet', default=True, action='store_true',
                        help='True: glimpse sensor is convnet, False: fully-connected')

    parser.add_argument('--p_labels', '-pl', type=float, default=1,
                        help='Fraction of labeled data')
    FLAGS, _ = parser.parse_known_args()

    time_str = datetime.now().strftime('%H%M%S')

    if FLAGS.model == 'ram':
        from config import Config
    elif FLAGS.model == 'dram':
        from config_dram import Config
    elif FLAGS.model == 'dram_loc':
        from config_dram import Config
    else:
        print 'Unknown model {}'.format(FLAGS.model)
        exit()

    # parameters
    config = Config()
    # n_steps = config.step

    # parameters
    config.use_context = FLAGS.use_context
    config.convnet = FLAGS.convnet
    config.p_labels = FLAGS.p_labels
    config.weighted_loss = FLAGS.weighted_loss
    config.balance = FLAGS.balance

    # log directory
    FLAGS.logdir = "./experiments/task={}{}x{}_model={}_conv={}_n_glimpses={}_fovea={}x{}_std={}_{}_context={}_lr={}-{}_p_labels={}_6".format(
        FLAGS.task, config.new_size, config.new_size,
        FLAGS.model, config.convnet, config.num_glimpses, config.glimpse_size, config.glimpse_size,
        config.loc_std, time_str, config.use_context, config.lr_start, config.lr_min,
        config.p_labels)

    print '\n\nFlags: {}\n\n'.format(FLAGS)
    # ------------------------------

    # data
    data = chest_xray(config)
    # config.w_plus = np.array([1., 1., 1., 1., 1.]).astype(np.float32)
    # config.w_plus = 1.
    config.w_plus = (data.y_train.shape[0] - np.sum(data.y_train, axis=0)) / (np.sum(data.y_train, axis=0))

    # init model
    config.sensor_size = config.glimpse_size ** 2 * config.n_patches
    config.N = data.x_train.shape[0]  # number of training examples

    if FLAGS.model == 'ram':
        print '\n\n\nTraining RAM\n\n\n'
        model = RAM(config, logdir=FLAGS.logdir)
    elif FLAGS.model == 'dram':
        print '\n\n\nTraining DRAM\n\n\n'
        model = DRAM(config, logdir=FLAGS.logdir)
    elif FLAGS.model == 'dram_loc':
        print '\n\n\nTraining DRAM with location ground truth\n\n\n'
        model = DRAMl(config, logdir=FLAGS.logdir)
    else:
        print 'Unknown model {}'.format(FLAGS.model)
        exit()

    # load if specified
    if FLAGS.load is not None:
        model.load(FLAGS.load)
        model.visualize(config=[], data=data, task={'variant': 'cluttered', 'width': 60, 'n_distractors': 4},
                        plot_dir='.', N=49, seed=None)
    # display # parameters
    model.count_params()

    # train
    model.train(data, FLAGS.task)
    model.evaluate(data=data, task=FLAGS.task)
