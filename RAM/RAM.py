"""
Implements Recurrent Attention Model (RAM) based on [1]

[1] Mnih et al. 2014. Recurrent Attention Model.
"""

import tensorflow as tf
import numpy as np
import os

import time

from Logger import Logger
from read_chestxray import *
from data_generator import *
from src.utils import *
from src.fig import plot_glimpses, plot_trajectories

# tensorflow version switch
rnn_cell = tf.contrib.rnn
seq2seq = tf.contrib.legacy_seq2seq


class RAM(object):
    """Implements RAM model."""

    def __init__(self, config, logdir='.'):

        # parameters
        self.config = config
        self.logdir = logdir

        # input placeholders
        self.images_ph = tf.placeholder(tf.float32,
                                        [None, self.config.new_size, self.config.new_size, self.config.num_channels])
        self.labels_ph = tf.placeholder(tf.float32, [None, self.config.num_classes])
        self.N = tf.shape(self.images_ph)[0]  # number of examples

        if self.config.convnet:
            print 'Glimpse sensor is Convnet.'
            from ConvNet import GlimpseNetwork, LocNet          # glimpse net is conv net
        else:
            print 'Glimpse sensor is fully connected.'
            from GlimpseNetwork import GlimpseNetwork, LocNet   # glimpse net if fully connected

        # glimpse network
        with tf.variable_scope('glimpse_net'):
            self.gl = GlimpseNetwork(self.config, self.images_ph)
        with tf.variable_scope('loc_net'):
            self.loc_net = LocNet(self.config)

        # initial glimpse
        # self.init_loc           = tf.random_uniform((self.N, 2), minval=-0.5, maxval=0.5)
        self.init_loc = tf.zeros(shape=[self.N, 2], dtype=tf.float32, )
        # self.init_loc = [0, -0.25] * tf.ones(shape=[self.N, 2], dtype=tf.float32)

        self.init_glimpse = self.gl(self.init_loc)
        self.inputs = [self.init_glimpse]
        self.inputs.extend([0] * (self.config.num_glimpses - 1))

        # ------- Core: recurrent network -------
        self.loc_mean_arr = []
        self.sampled_loc_arr = []
        self.glimpses = []
        self.glimpses.append(self.gl.glimpse_img)

        def get_next_input(output, i):
            """Samples next glimpse location."""
            loc, loc_mean = self.loc_net(output)  # takes hidden RNN state and produces next location
            gl_next = self.gl(loc)

            # for visualization
            self.loc_mean_arr.append(loc_mean)
            self.sampled_loc_arr.append(loc)
            self.glimpses.append(self.gl.glimpse_img)

            return gl_next

        self.lstm_cell = rnn_cell.LSTMCell(self.config.cell_size, state_is_tuple=True, activation=tf.nn.relu)
        self.init_state = self.lstm_cell.zero_state(self.N, tf.float32)

        # output: list of num_glimpses + 1
        self.outputs, _ = seq2seq.rnn_decoder(self.inputs, self.init_state, self.lstm_cell, loop_function=get_next_input)
        get_next_input(self.outputs[-1], 0)
        # ---------------------------------------

        # time independent baselines
        with tf.variable_scope('baseline'):
            w_baseline = weight_variable((self.config.cell_output_size, 1))
            b_baseline = bias_variable((1,))
        baselines = []

        for t, output in enumerate(self.outputs):  # ignore initial state
            baseline_t = tf.nn.xw_plus_b(output, w_baseline, b_baseline)
            baseline_t = tf.squeeze(baseline_t)
            baselines.append(baseline_t)

        # outputs for each glimpse (t)
        baselines = tf.stack(baselines)  # [timesteps, batch_size]
        self.baselines = tf.transpose(baselines)  # [batch_size, timesteps]

        # Take the last step only.
        self.output = self.outputs[-1]  # [batch size x cell_output_size]

        # ---- for visualizations ----
        self.sampled_locations = tf.concat(self.sampled_loc_arr, axis=0)
        self.mean_locations = tf.concat(self.loc_mean_arr, axis=0)

        # self.sampled_locs = tf.reshape(self.sampled_locs, (self.batch_size, self.glimpses, 2))
        self.sampled_locations = tf.reshape(self.sampled_locations, (self.config.num_glimpses, self.N, 2))
        self.sampled_locations = tf.transpose(self.sampled_locations, [1, 0, 2])

        self.mean_locations = tf.reshape(self.mean_locations, (self.config.num_glimpses, self.N, 2))
        self.mean_locations = tf.transpose(self.mean_locations, [1, 0, 2])

        prefix = tf.expand_dims(self.init_loc, 1)
        self.sampled_locations = tf.concat([prefix, self.sampled_locations], axis=1)
        self.mean_locations = tf.concat([prefix, self.mean_locations], axis=1)

        self.glimpses = tf.stack(self.glimpses, axis=1)
        # -----------------------------

        # classification network
        with tf.variable_scope('classification'):
            w_logit = weight_variable((self.config.cell_output_size, self.config.num_classes))
            b_logit = bias_variable((self.config.num_classes,))
            # self.logits = tf.reshape(tf.nn.xw_plus_b(self.output, w_logit, b_logit), [-1])
            self.logits = tf.nn.xw_plus_b(self.output, w_logit, b_logit)
            self.softmax = tf.nn.sigmoid(self.logits)  # [batch_size x n_classes]

            # class probabilities for each glimpse
            self.class_prob_arr = []

            for op in self.outputs:
                self.glimpse_logit = tf.stop_gradient(tf.nn.xw_plus_b(op, w_logit, b_logit))
                self.class_prob_arr.append(tf.nn.sigmoid(self.glimpse_logit))

            self.class_prob_arr = tf.stack(self.class_prob_arr, axis=1)
            # [batch_size x num_glimpses x n_classes]
        # Losses/reward

        # cross-entropy
        if self.config.weighted_loss:
            # xent = tf.nn.weighted_cross_entropy_with_logits(targets=self.labels_ph, logits=self.logits,
            #                                                 pos_weight=self.config.w_plus)
            self.xent = self.cross_entropy_loss(self.config.w_plus, weighted_loss=self.config.weighted_loss)
        else:
            xent = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels_ph, logits=self.logits)
            self.xent = tf.reduce_mean(xent)
        self.pred_labels = tf.cast(tf.round(self.softmax), tf.float32)

        # REINFORCE: 0/1 reward
        self.reward = tf.reduce_mean(tf.cast(tf.equal(self.pred_labels, self.labels_ph), tf.float32), axis=1)

        self.rewards = tf.expand_dims(self.reward, 1)  # [batch_sz, 1]
        self.rewards = tf.tile(self.rewards, (1, self.config.num_glimpses))  # [batch_sz, timesteps]
        self.logll = loglikelihood(self.loc_mean_arr, self.sampled_loc_arr, self.config.loc_std)
        self.advs = self.rewards - tf.stop_gradient(self.baselines)
        self.logllratio = tf.reduce_mean(self.logll * self.advs)
        self.reward = tf.reduce_mean(self.reward)

        self.baselines_mse = tf.reduce_mean(tf.square((self.rewards - self.baselines)))  # original
        self.var_list = tf.trainable_variables()

        # hybrid loss
        self.loss = -self.logllratio + self.xent + self.baselines_mse  # `-` to minimize
        self.grads = tf.gradients(self.loss, self.var_list)
        self.grads, _ = tf.clip_by_global_norm(self.grads, self.config.max_grad_norm)

        # set up optimization
        self.setup_optimization()

        # session
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def cross_entropy_loss(self, w_plus, weighted_loss=True):
        """
         Calculates the cross-entropy loss function for the given parameters.
        :param labels_tensor: Tensor of correct predictions of size [batch_size, numClasses]
        :param logits_tensor: Predicted scores (logits) by the model.
                It should have the same dimensions as labels_tensor
        :return: Cross-entropy loss value over the samples of the current batch
        """
        if weighted_loss:
            labels_series = tf.unstack(self.labels_ph, axis=1)
            logits_series = tf.unstack(self.logits, axis=1)
            w_plus = w_plus.astype(np.float32)
            losses_list = [tf.nn.weighted_cross_entropy_with_logits(logits=logits, targets=labels, pos_weight=w_p)
                           for (logits, labels, w_p) in
                           zip(logits_series, labels_series, tf.split(w_plus, self.config.num_classes))]
            diff = tf.stack(losses_list, axis=1)
        else:
            diff = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.labels_ph)
        loss = tf.reduce_mean(diff)
        return loss

    def setup_optimization(self, training_steps_per_epoch=None):
        """Set up optimzation operators."""

        # learning rate
        self.global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        self.training_steps_per_epoch = self.config.N // self.config.batch_size
        # print 'Training steps / epoch {}'.format(self.training_steps_per_epoch)

        self.starter_learning_rate = self.config.lr_start
        # decay per training epoch
        self.learning_rate = tf.train.exponential_decay(self.starter_learning_rate,
                                                        self.global_step,
                                                        self.training_steps_per_epoch,
                                                        0.97,
                                                        staircase=True)
        self.learning_rate = tf.maximum(self.learning_rate, self.config.lr_min)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = self.optimizer.apply_gradients(zip(self.grads, self.var_list), global_step=self.global_step)

    def setup_logger(self):
        """Creates log directory and initializes logger."""
        loc_net1 = [v for v in tf.global_variables() if v.name == "loc_net/Variable:0"]

        self.summary_ops = {
            'reward': tf.summary.scalar('reward', self.reward),
            'hybrid_loss': tf.summary.scalar('hybrid_loss', self.loss),
            'cross_entropy': tf.summary.scalar('cross_entropy', self.xent),
            'baseline_mse': tf.summary.scalar('baseline_mse', self.baselines_mse),
            'logllratio': tf.summary.scalar('logllratio', self.logllratio),
            'loc_net1': tf.summary.histogram('loc_net1', loc_net1),
            'glimpses': tf.summary.image('glimpses',
                                         tf.reshape(self.glimpses,
                                                    [-1, self.config.glimpse_size, self.config.glimpse_size,
                                                     self.config.num_channels]),
                                         max_outputs=20)
        }
        self.eval_ops = {
            'reward': self.reward,
            'hybrid_loss': self.loss,
            'cross_entropy': self.xent,
            'baseline_mse': self.baselines_mse,
            'logllratio': self.logllratio,
            'lr': self.learning_rate
        }
        self.logger = Logger(self.logdir, sess=self.session, summary_ops=self.summary_ops,
                             global_step=self.global_step, eval_ops=self.eval_ops,
                             n_verbose=self.config.n_verbose, var_list=self.var_list)

    def train(self, data=[], task='mnist'):
        """Trains RAM model and logs statistics.

        Args:
            data    -- data set object (.train, .test, .validation), cf mnist
            task    -- str ['mnist','translated','cluttered']
            data    -- data set object (.train, .test, .validation), cf mnist
        """
        # verbose
        print '\n\n\n------------ Starting training ------------  \nTask: {} -- {}x{} with {} distractors\n' \
              'Model: {} patches, {} glimpses, glimpse size {}x{}\n\n\n'.format(
            task, self.config.new_size, self.config.new_size, self.config.n_distractors,
            self.config.n_patches, self.config.num_glimpses, self.config.glimpse_size, self.config.glimpse_size
        )

        self.task = task
        self.setup_logger()  # add logger
        step_count = int(len(data.x_train) / self.config.batch_size)
        for epoch in range(self.config.num_epoch):
            data.x_train, data.y_train = randomize(data.x_train, data.y_train)
            for step in range(step_count):
                # t = time.time()
                start = step * self.config.batch_size
                end = (step + 1) * self.config.batch_size
                images, labels = get_next_batch(data.x_train, data.y_train, start, end)
                images = images.reshape((-1, self.config.original_size, self.config.original_size, 1))

                images = add_noise(images, mode='pepper', amount=0.05)

                # duplicate M times, see Eqn (2)
                images = np.tile(images, [self.config.M, 1, 1, 1])
                labels = np.tile(labels, [self.config.M, 1])
                self.loc_net.sampling = True

                # training step
                feed_dict = {self.images_ph: images, self.labels_ph: labels}
                self.session.run(self.train_op, feed_dict=feed_dict)

                # log
                self.logger.step = epoch * step_count + step
                self.logger.log('train', feed_dict=feed_dict)
                # print('iteration time: {}'.format(time.time() - t))

                # evaluation on test/validation
                # if i and i % (2 * self.training_steps_per_epoch) == 0:
                # if step and epoch * step_count + step % 1000 == 0:
                # if step and step % 2 == 0:
            # save model
            self.logger.save(epoch)
            print '\n==== Evaluation: (Epoch #{}) ===='.format(epoch)
            self.evaluate(data, task=self.task)

    def evaluate(self, data=[], task='mnist'):
        """Returns accuracy of current model.

        Returns:
            test_accuracy, validation_accuracy

        """
        return evaluate(self.session, self.images_ph, self.labels_ph, self.softmax, self.loss, data, self.config, task)

    def load(self, checkpoint_dir):
        """Restores model from <<checkpoint_dir>>. Assumes sub-folder 'checkpoints' in directory."""

        folder = os.path.join(checkpoint_dir, 'checkpoints/')
        print '\nLoading model from <<{}>>.\n'.format(folder)

        self.saver = tf.train.Saver(self.var_list)

        ckpt = tf.train.get_checkpoint_state(folder)

        if ckpt and ckpt.model_checkpoint_path:
            print ckpt
            self.saver.restore(self.session, ckpt.model_checkpoint_path)

    def visualize(self, config=[], data=[], task={'variant': 'mnist', 'width': 60, 'n_distractors': 4},
                  plot_dir='.', N=16, seed=None):
        """Given a saved model visualizes inference.

        Args:
                config          params
                data            data object (cf mnist)
                task            (dict) parameters for task to evaluate on

                N               (int) number of plots
                seed            (int) random if 'None', seed='seed' o.w.
        """
        print '\n\nGenerating visualizations ....',

        np.random.seed(seed)

        # evaluation task
        self.loc_net.sampling = False

        # config.new_size         = task['width']
        # config.n_distractors    = task['n_distractors']

        # data
        X = data.x_train.reshape((-1, 256, 256, 1))
        labels = data.y_train

        # test model
        # if task['variant'] == 'translated':
        #     X = translate(X, width=task['width'], height=task['width'])
        # elif task['variant'] == 'cluttered':
        #     X = clutter(X, X, width=task['width'], height=task['width'], n_patches=task['n_distractors'])
        # else:
        #     print 'Using original MNIST data.'

        # sample random subset of data
        idx = np.random.permutation(X.shape[0])[:N]

        X, Y = X[idx], labels[idx]

        # data for plotting
        feed_dict = {self.images_ph: X, self.labels_ph: Y}
        fetches = [self.glimpses, self.sampled_locations, self.mean_locations, self.pred_labels, self.class_prob_arr]

        results = self.session.run(fetches, feed_dict=feed_dict)
        glimpse_images, sampled_locations, mean_locations, pred_labels, probs = results

        # glimpses
        plot_glimpses(config=self.config, glimpse_images=glimpse_images, pred_labels=pred_labels, probs=[],
                      sampled_loc=mean_locations,
                      X=X, labels=Y, file_name=os.path.join(plot_dir, 'glimpses_mean'))

        plot_trajectories(config=self.config, locations=mean_locations,
                          X=X, labels=Y, pred_labels=pred_labels, file_name=os.path.join(plot_dir, 'trajectories'))

    def count_params(self):
        """Returns number of trainable parameters."""
        return count_parameters(self.session)
