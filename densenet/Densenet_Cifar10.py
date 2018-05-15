import tensorflow as tf
from tflearn.layers.conv import global_avg_pool
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib.framework import arg_scope
# from cifar10 import *
import datetime
import h5py
import numpy as np
import os
# from sklearn.cross_validation import train_test_split
from sklearn.metrics import precision_recall_fscore_support
# from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
import cPickle as pickle


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="2, 3"
# Hyperparameter
growth_k = 32
nb_block = 4 # how many (dense block + Transition Layer) ?
init_learning_rate = 1e-4
epsilon = 1e-4 # AdamOptimizer epsilon
dropout_rate = 0.2

# Momentum Optimizer will use
nesterov_momentum = 0.9
weight_decay = 5e-4

# Label & batch_size
batch_size = 2

# batch_size * iteration = data_set_number
freq = 1000

total_epochs = 40

def conv_layer(input, filter, kernel, stride=1, layer_name="conv"):
    with tf.name_scope(layer_name):
        network = tf.layers.conv2d(inputs=input, use_bias=False, filters=filter, kernel_size=kernel, strides=stride, padding='SAME')
        return network

def Global_Average_Pooling(x, stride=1):
    """
    width = np.shape(x)[1]
    height = np.shape(x)[2]
    pool_size = [width, height]
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride) # The stride value does not matter
    It is global average pooling without tflearn
    """

    return global_avg_pool(x, name='Global_avg_pooling')
    # But maybe you need to install h5py and curses or not


def Batch_Normalization(x, training, scope):
    with arg_scope([batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True) :
        return tf.cond(training,
                       lambda : batch_norm(inputs=x, is_training=training, reuse=None),
                       lambda : batch_norm(inputs=x, is_training=training, reuse=True))

def Drop_out(x, rate, training) :
    return tf.layers.dropout(inputs=x, rate=rate, training=training)

def Relu(x):
    return tf.nn.relu(x)

def Average_pooling(x, pool_size=[2,2], stride=2, padding='VALID'):
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)


def Max_Pooling(x, pool_size=[3,3], stride=2, padding='VALID'):
    return tf.layers.max_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)

def Concatenation(layers) :
    return tf.concat(layers, axis=3)

def Linear(x) :
    return tf.layers.dense(inputs=x, units=class_num, name='linear')

# def Evaluate(sess, test_iteration):
#     # test_acc = 0.0
#     test_loss = 0.0
#     test_pre_index = 0
#     add = batch_size
#     test_labels = test_preds = np.zeros((0, class_num))
#
#     for it in range(test_iteration):
#         test_batch_x = x_test[test_pre_index: test_pre_index + add]
#         test_batch_y = y_test[test_pre_index: test_pre_index + add]
#         test_pre_index = test_pre_index + add
#
#         test_feed_dict = {
#             x: test_batch_x,
#             label: test_batch_y,
#             # learning_rate: epoch_learning_rate,
#             training_flag: False
#         }
#
#         loss_, preds_ = sess.run([cost, preds], feed_dict=test_feed_dict)
#
#         test_loss += loss_
#         test_preds = np.concatenate((test_preds, preds_))
#         test_labels = np.concatenate((test_labels, test_batch_y))
#     # auroc_0 = roc_auc_score(y_test[:, 0], test_preds[:, 0])
#     # auroc_1 = roc_auc_score(y_test[:, 1], test_preds[:, 1])
#     # test_preds = np.round(test_preds)
#     loss = test_loss/test_iteration
#     # acc = test_acc/test_iteration
#     auroc = auroc_generator(test_labels, test_preds)
#     precision, recall, f1 = prf_generator(test_labels, test_preds)
#
#     return auroc, precision, recall, f1, loss

def auroc_generator(labels, preds):
    auroc = []
    for i in range(14):
        auroc.append(roc_auc_score(labels[:, i], preds[:, i]))
    return np.asarray(auroc)

def prf_generator(labels, preds):
    precision = []
    recall = []
    f1_score = []
    for i in range(14):
        p,r,f,_ = precision_recall_fscore_support(labels[:, i], np.round(preds[:, i]), pos_label=1, average='binary')
        precision.append(p)
        recall.append(r)
        f1_score.append(f)
    return np.asarray(precision), np.asarray(recall), np.asarray(f1_score)

class DenseNet():
    def __init__(self, x, nb_blocks, filters, training):
        self.nb_blocks = nb_blocks
        self.filters = filters
        self.training = training
        self.model, self.features = self.Dense_net(x)


    def bottleneck_layer(self, x, scope):
        # print(x)
        with tf.name_scope(scope):
            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
            x = Relu(x)
            x = conv_layer(x, filter=4 * self.filters, kernel=[1,1], layer_name=scope+'_conv1')
            x = Drop_out(x, rate=dropout_rate, training=self.training)

            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch2')
            x = Relu(x)
            x = conv_layer(x, filter=self.filters, kernel=[3,3], layer_name=scope+'_conv2')
            x = Drop_out(x, rate=dropout_rate, training=self.training)

            # print(x)

            return x

    def transition_layer(self, x, scope):
        with tf.name_scope(scope):
            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
            x = Relu(x)
            x = conv_layer(x, filter=self.filters, kernel=[1,1], layer_name=scope+'_conv1')
            x = Drop_out(x, rate=dropout_rate, training=self.training)
            x = Average_pooling(x, pool_size=[2,2], stride=2)

            return x

    def dense_block(self, input_x, nb_layers, layer_name):
        with tf.name_scope(layer_name):
            layers_concat = list()
            layers_concat.append(input_x)

            x = self.bottleneck_layer(input_x, scope=layer_name + '_bottleN_' + str(0))

            layers_concat.append(x)

            for i in range(nb_layers - 1):
                x = Concatenation(layers_concat)
                x = self.bottleneck_layer(x, scope=layer_name + '_bottleN_' + str(i + 1))
                layers_concat.append(x)

            x = Concatenation(layers_concat)

            return x

    def Dense_net(self, input_x):
        x = conv_layer(input_x, filter=2 * self.filters, kernel=[7,7], stride=2, layer_name='conv0')
        # x = Max_Pooling(x, pool_size=[3,3], stride=2)


        """
        for i in range(self.nb_blocks) :
            # 6 -> 12 -> 48
            x = self.dense_block(input_x=x, nb_layers=4, layer_name='dense_'+str(i))
            x = self.transition_layer(x, scope='trans_'+str(i))
        """


        x = self.dense_block(input_x=x, nb_layers=6, layer_name='dense_1')
        x = self.transition_layer(x, scope='trans_1')

        x_mine = self.dense_block(input_x=x, nb_layers=12, layer_name='dense_2')
        x = self.transition_layer(x_mine, scope='trans_2')

        x = self.dense_block(input_x=x, nb_layers=24, layer_name='dense_3')
        x = self.transition_layer(x, scope='trans_3')

        x = self.dense_block(input_x=x, nb_layers=16, layer_name='dense_final')



        # 100 Layer
        x = Batch_Normalization(x, training=self.training, scope='linear_batch')
        x = Relu(x)
        x = Global_Average_Pooling(x)
        x = flatten(x)
        x = Linear(x)


        # x = tf.reshape(x, [-1, 10])
        return x, x_mine


# h5f_train = h5py.File('/home/cougarnet.uh.edu/amobiny/Desktop/chest_xray/data/chest256_train_801010_no_normal.h5', 'r')
# x_train = h5f_train['X_train'][:]
# y_train = h5f_train['Y_train'][:]
# h5f_train.close()
#
# h5f_test = h5py.File('/home/cougarnet.uh.edu/amobiny/Desktop/chest_xray/data/chest256_val_801010_no_normal.h5', 'r')
# x_test = h5f_test['X_val'][:]
# y_test = h5f_test['Y_val'][:]
# h5f_test.close()
with open('/home/cougarnet.uh.edu/amobiny/Desktop/Radiologist_Gaze/code/resized_gaze_img_dict.pkl', 'rb') as inputs:
    img_sub_dict = pickle.load(inputs)
print('Input Loaded')
image_name_list = []
input_image = np.zeros([0, 256, 256])
for image_name, value in img_sub_dict.items():
    image_name_list.append(image_name)
    input_image = np.concatenate((np.expand_dims(value[0], axis=0), input_image), axis=0)


mean = 127.1612
std = 63.5096

input_image = (input_image - mean)/std
input_image = np.reshape(input_image, [-1, 256, 256, 1])

img_size = 256
img_channels = 1
class_num = 14


# image_size = 32, img_channels = 3, class_num = 10 in cifar10
x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, img_channels])
label = tf.placeholder(tf.float32, shape=[None, class_num])

training_flag = tf.placeholder(tf.bool)

# learning_rate = tf.placeholder(tf.float32, name='learning_rate')

ext_feat = DenseNet(x=x, nb_blocks=nb_block, filters=growth_k, training=training_flag).features
# l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])

# w_plus = (y_train.shape[0] - np.sum(y_train, axis=0)) / (np.sum(y_train, axis=0))
# cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=logits))
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logits))
# cost = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets=label, logits=logits, pos_weight= w_plus))
#
# correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(label, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#
# preds = tf.nn.sigmoid(logits)
# preds = tf.round(preds)





"""
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=nesterov_momentum, use_nesterov=True)
train = optimizer.minimize(cost + l2_loss * weight_decay)

In paper, use MomentumOptimizer
init_learning_rate = 0.1

but, I'll use AdamOptimizer
"""

# optimizer = tf.train.AdamOptimizer(learning_rate=init_learning_rate, epsilon=epsilon)
# train = optimizer.minimize(cost)

saver = tf.train.Saver(tf.global_variables())

# iteration = x_train.shape[0]/batch_size
conditions = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
              'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
with tf.Session() as sess:
    currentDT = datetime.datetime.now()
    ckpt = tf.train.get_checkpoint_state('./model/2018-04-18_16-19-15_unwt_loss/')
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())
    merged = tf.summary.merge_all()
    a = sess.run(ext_feat, feed_dict={x: input_image, label: np.random.rand(input_image.shape[0], 14), training_flag : False})
    print()







    #
    #
    #
    #
    #
    # train_writer = tf.summary.FileWriter('./logs/'+currentDT.strftime("%Y-%m-%d_%H-%M-%S")+'/train/', sess.graph)
    # test_writer = tf.summary.FileWriter('./logs/'+currentDT.strftime("%Y-%m-%d_%H-%M-%S")+'/test/')
    # epoch_learning_rate = init_learning_rate
    # all_batch_preds = all_batch_labels = np.zeros((0, 14))
    # for epoch in range(1, total_epochs + 1):
    #     if epoch == (total_epochs * 0.5) or epoch == (total_epochs * 0.75):
    #         epoch_learning_rate = epoch_learning_rate / 10
    #
    #     pre_index = 0
    #     train_acc = 0.0
    #     train_loss = 0.0
    #     preds_frq_steps = np.zeros((batch_size*freq, class_num))
    #     index = 0
    #
    #
    #     for step in range(1, iteration + 1):
    #         # if pre_index+batch_size < 50000 :
    #         batch_x = x_train[pre_index : pre_index+batch_size]
    #         batch_y = y_train[pre_index : pre_index+batch_size]
    #         # else :
    #         #     batch_x = train_x[pre_index : ]
    #         #     batch_y = train_y[pre_index : ]
    #         #
    #         # batch_x = data_augmentation(batch_x)
    #
    #         train_feed_dict = {
    #             x: batch_x,
    #             label: batch_y,
    #             # learning_rate: epoch_learning_rate,
    #             training_flag : True
    #         }
    #
    #         _, batch_loss = sess.run([train, cost], feed_dict = train_feed_dict)
    #         # train_writer.add_summary(summary, iteration * (epoch - 1) + step)
    #         # batch_acc = accuracy.eval(feed_dict = train_feed_dict)
    #         batch_preds = preds.eval(feed_dict = train_feed_dict)
    #         all_batch_labels = np.concatenate((all_batch_labels, batch_y))
    #         all_batch_preds = np.concatenate((all_batch_preds, batch_preds))
    #
    #         train_loss += batch_loss
    #         # train_acc += batch_acc
    #         pre_index += batch_size
    #         preds_frq_steps[index*batch_size : (index+1)*batch_size, :] = batch_preds
    #         index += 1
    #         if step%freq == 0 :
    #             loss = train_loss/freq # average loss
    #             # acc = train_acc/freq # average accuracy
    #             auroc = auroc_generator(all_batch_labels, all_batch_preds)
    #             precision, recall, f1 = prf_generator(all_batch_labels, all_batch_preds)
    #             all_batch_preds = all_batch_labels = np.zeros((0, 14))
    #             train_loss = 0.0
    #             # train_acc = 0.0
    #             # train_summary = tf.Summary(value=[tf.Summary.Value(tag='train_loss', simple_value=loss),
    #             #                                   tf.Summary.Value(tag='train_accuracy', simple_value=acc)])
    #
    #             # labels = y_train[(step - freq)*batch_size : step*batch_size, :]
    #             # cond_count = np.sum(labels[:, 1])
    #             # if cond_count>0:
    #             #     auroc_0 = roc_auc_score(labels[:, 0], preds_frq_steps[:, 0])
    #             #     auroc_1 = roc_auc_score(labels[:, 1], preds_frq_steps[:, 1])
    #             # else:
    #             #     auroc_1 = auroc_0 = 0
    #             # preds_frq_steps = np.round(preds_frq_steps)
    #
    #             # precision, recall, f1, _ = precision_recall_fscore_support(labels, preds_frq_steps)
    #
    #             # test_acc, test_loss, test_summary = Evaluate(sess)
    #             train_summary = tf.Summary(value=[tf.Summary.Value(tag='Mean_loss', simple_value=loss)])
    #             train_writer.add_summary(train_summary, iteration*(epoch-1) + step)
    #
    #             for cond in range(class_num):
    #                 with tf.name_scope('AUROC'):
    #                     summary_tr = tf.Summary(
    #                         value=[tf.Summary.Value(tag='AUROC_' + conditions[cond],
    #                                                 simple_value=auroc[cond])])
    #                     train_writer.add_summary(summary_tr, iteration*(epoch-1) + step)
    #                 with tf.name_scope('Precision'):
    #                     summary_tr = tf.Summary(
    #                         value=[tf.Summary.Value(tag='Precision_' + conditions[cond],
    #                                                 simple_value=precision[cond])])
    #                     train_writer.add_summary(summary_tr, iteration*(epoch-1) + step)
    #                 with tf.name_scope('Recall'):
    #                     summary_tr = tf.Summary(
    #                         value=[tf.Summary.Value(tag='Recall_' + conditions[cond],
    #                                                 simple_value=recall[cond])])
    #                     train_writer.add_summary(summary_tr, iteration*(epoch-1) + step)
    #                 with tf.name_scope('F1_score'):
    #                     summary_tr = tf.Summary(
    #                         value=[tf.Summary.Value(tag='F1_' + conditions[cond],
    #                                                 simple_value=f1[cond])])
    #                     train_writer.add_summary(summary_tr, iteration*(epoch-1) + step)
    #
    #             summary_tr = tf.Summary(value=[tf.Summary.Value(tag='AUROC/Mean_AUROC',
    #                                                             simple_value=np.mean(auroc)),
    #                                            tf.Summary.Value(tag='Precision/Mean_Precision',
    #                                                             simple_value=np.mean(precision)),
    #                                            tf.Summary.Value(tag='Recall/Mean_Recall',
    #                                                             simple_value=np.mean(recall)),
    #                                            tf.Summary.Value(tag='F1/Mean_F1',
    #                                                             simple_value=np.mean(f1))
    #                                            ])
    #             train_writer.add_summary(summary_tr, iteration * (epoch - 1) + step)
    #
    #             # summary = sess.run(merged, feed_dict=train_feed_dict)
    #             # train_writer.add_summary(summary, iteration * (epoch - 1) + step)
    #             # train_writer.add_summary(summary=summary, global_step=step*epoch)
    #             # summary_writer.add_sum
    #             # mary(summary=test_summary, global_step=step)
    #             # summary_writer.flush()
    #             # preds_frq_steps = np.zeros((batch_size*freq, class_num))
    #             index = 0
    #             print("Atlc\tCrdmg\tEffus\tInflt\tMass\tNodle\tPnum\tPntrx\tConsd"
    #                   "\tEdma\tEmpys\tFbrss\tTkng\tHrna\t|Avg.\t|Loss\t|Step\t|Epoch")
    #             print(
    #             '{0:.2f}\t|{1:.2f}\t|{2:.2f}\t|{3:.2f}\t|{4:.2f}\t|{5:.2f}\t|{6:.2f}\t|{7:.2f}\t|{8:.2f}\t|{9:.2f}\t'
    #             '|{10:.2f}\t|{11:.2f}\t|{12:.2f}\t|{13:.2f}\t|{14:.2f}\t|{15:.2f}\t|{16}\t|{17}'
    #             .format(auroc[0], auroc[1], auroc[2], auroc[3], auroc[4], auroc[5], auroc[6], auroc[7], auroc[8],
    #                     auroc[9], auroc[10], auroc[11], auroc[12], auroc[13], np.mean(auroc), loss, step, epoch))
    #
    #
    #             # with open('logs.txt', 'a') as f :
    #             #     f.write(line)
    #     test_auroc, test_precision, test_recall, test_f1, test_loss = Evaluate(sess, y_test.shape[0]/batch_size)
    #     train_summary = tf.Summary(value=[tf.Summary.Value(tag='Mean_loss', simple_value=test_loss)])
    #     test_writer.add_summary(train_summary, iteration * (epoch - 1) + step)
    #     for cond in range(class_num):
    #         with tf.name_scope('AUROC'):
    #             summary_tr = tf.Summary(
    #                 value=[tf.Summary.Value(tag='AUROC_' + conditions[cond],
    #                                         simple_value=test_auroc[cond])])
    #             test_writer.add_summary(summary_tr, iteration * (epoch - 1) + step)
    #         with tf.name_scope('Precision'):
    #             summary_tr = tf.Summary(
    #                 value=[tf.Summary.Value(tag='Precision_' + conditions[cond],
    #                                         simple_value=test_precision[cond])])
    #             test_writer.add_summary(summary_tr, iteration * (epoch - 1) + step)
    #         with tf.name_scope('Recall'):
    #             summary_tr = tf.Summary(
    #                 value=[tf.Summary.Value(tag='Recall_' + conditions[cond],
    #                                         simple_value=test_recall[cond])])
    #             test_writer.add_summary(summary_tr, iteration * (epoch - 1) + step)
    #         with tf.name_scope('F1_score'):
    #             summary_tr = tf.Summary(
    #                 value=[tf.Summary.Value(tag='F1_' + conditions[cond],
    #                                         simple_value=test_f1[cond])])
    #             test_writer.add_summary(summary_tr, iteration * (epoch - 1) + step)
    #
    #     summary_tr = tf.Summary(value=[tf.Summary.Value(tag='AUROC/Mean_AUROC',
    #                                                     simple_value=np.mean(test_auroc)),
    #                                    tf.Summary.Value(tag='Precision/Mean_Precision',
    #                                                     simple_value=np.mean(test_precision)),
    #                                    tf.Summary.Value(tag='Recall/Mean_Recall',
    #                                                     simple_value=np.mean(test_recall)),
    #                                    tf.Summary.Value(tag='F1/Mean_F1',
    #                                                     simple_value=np.mean(test_f1))
    #                                    ])
    #     test_writer.add_summary(summary_tr, iteration * (epoch - 1) + step)
    #     print('---------------------------------------Validation----------------------------------------------')
    #     print("Atlc\tCrdmg\tEffus\tInflt\tMass\tNodle\tPnum\tPntrx\tConsd"
    #           "\tEdma\tEmpys\tFbrss\tTkng\tHrna\t|Avg.\t|Loss\t|Step")
    #     print(
    #         '{0:.2f}\t|{1:.2f}\t|{2:.2f}\t|{3:.2f}\t|{4:.2f}\t|{5:.2f}\t|{6:.2f}\t|{7:.2f}\t|{8:.2f}\t|{9:.2f}\t'
    #         '|{10:.2f}\t|{11:.2f}\t|{12:.2f}\t|{13:.2f}\t|{14:.2f}\t|{15:.2f}\t|{16}'
    #             .format(test_auroc[0], test_auroc[1], test_auroc[2], test_auroc[3], test_auroc[4], test_auroc[5], test_auroc[6], test_auroc[7], test_auroc[8],
    #                     test_auroc[9], test_auroc[10], test_auroc[11], test_auroc[12], test_auroc[13], np.mean(test_auroc), test_loss, step))
    #     # test_cond_count = np.sum(y_test[:,1])
    #     # line = "epoch: %d/%d, step: %d, \ntest_loss: %.4f, test_acc: %.4f, test_precision_cond: %.4f, test_precision_no_cond: %.4f, test_recall_cond: %.4f, test_recall_no_cond: %.4f, test_auroc:%.4f," \
    #     #        " #conditions: %d/%d \n" % (
    #     #     epoch, total_epochs, step, test_loss, test_acc, precision[1], precision[0], recall[1], recall[0], auroc_1, test_cond_count, y_test.shape[0])
    #     # print(line)
    #
    #     save_path = './model/'+currentDT.strftime("%Y-%m-%d_%H-%M-%S")
    #     if not os.path.exists(save_path):
    #         os.makedirs(save_path)
    #     saver.save(sess=sess, save_path= save_path +'/densenet.ckpt')
    # train_writer.close()
    # test_writer.close()