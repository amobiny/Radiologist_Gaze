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
# batch_size = 2

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
batch_size = 2


def get_next_batch(x, start, end):
    x_batch = x[start:end]
    return x_batch


# image_size = 32, img_channels = 3, class_num = 10 in cifar10
x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, img_channels])
label = tf.placeholder(tf.float32, shape=[None, class_num])
training_flag = tf.placeholder(tf.bool)

ext_feat = DenseNet(x=x, nb_blocks=nb_block, filters=growth_k, training=training_flag).features

saver = tf.train.Saver(tf.global_variables())
all_feats = np.zeros((0, 64, 64, 416))
conditions = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
              'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
with tf.Session() as sess:
    currentDT = datetime.datetime.now()
    ckpt = tf.train.get_checkpoint_state('./model/2018-04-18_16-19-15_unwt_loss/')
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    step_count = int(input_image.shape[0]) / batch_size
    for step in range(step_count):
        start = step * batch_size
        end = (step + 1) * batch_size
        X_batch = get_next_batch(input_image, start, end)
        a = sess.run(ext_feat, feed_dict={x: X_batch, label: np.random.rand(X_batch.shape[0], 14), training_flag: False})
        all_feats = np.concatenate((all_feats, a), axis=0)
    print()

# save features
h5f = h5py.File('features_from_densenet.h5', 'w')
h5f.create_dataset('all_feats', data=all_feats)
h5f.create_dataset('image_name_list', data=image_name_list)
h5f.close()
