import random
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve
from config import args
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


def classifier_model(x_input, y_input, n_estimators, max_depth, max_features, feat_importance=False, split=True):
    classifier = RandomForestClassifier(n_estimators=n_estimators,
                                        # max_depth=max_depth,
                                        oob_score=True,
                                        max_features=max_features)
    if split:
        x_train, y_train, x_test, y_test = train_test_split(x_input, y_input)
    else:
        x_test = x_train = x_input
        y_test = y_train = y_input
    classifier.fit(x_train, y_train)
    y_pred_test = classifier.predict(x_test)
    y_pred_train = classifier.predict(x_train)
    y_prob_test = classifier.predict_proba(x_test)
    y_prob_train = classifier.predict_proba(x_train)
    # accuracy = np.sum(np.equal(np.argmax(result, 1), np.argmax(y_test, 1)))/np.float(y_test.shape[0])
    accuracy_test = np.sum(np.equal(np.reshape(y_pred_test, (-1, 1)),
                                    np.reshape(y_test, (-1, 1)))) / np.float(y_pred_test.size)
    accuracy_train = np.sum(np.equal(np.reshape(y_pred_train, (-1, 1)),
                                     np.reshape(y_train, (-1, 1)))) / np.float(y_pred_train.size)
    auroc_train = roc_auc_score(y_train, y_prob_train[:, -1])
    auroc_test = roc_auc_score(y_test, y_prob_test[:, -1])
    p_test, r_test = precision_recall(y_test, y_pred_test)
    p_train, r_train = precision_recall(y_train, y_pred_train)
    if feat_importance:
        features_imp = classifier.feature_importances_
        print('Classifier Trained & Feature importance generated')
        # plot_precision_recall(y_test, y_prob_test)
        return accuracy_test, features_imp, classifier
    else:
        return accuracy_test, accuracy_train, auroc_test, auroc_train, p_test, r_test, p_train, r_train


def multi_run(x, y, count, split=True):
    accur_test, accur_train, auroc_test_all, auroc_train_all, p_test_all, r_test_all, p_train_all, r_train_all = [], [], [], [], [], [], [], []
    for i in range(count):
        acc_test, acc_train, auroc_test, auroc_train, p_test, r_test, p_train, r_train = classifier_model(x, y,
                                                                                                          args.n_estimators,
                                                                                                          args.max_depth,
                                                                                                          args.max_features,
                                                                                                          split=split)
        accur_test.append(acc_test)
        accur_train.append(acc_train)
        auroc_test_all.append(auroc_test)
        auroc_train_all.append(auroc_train)
        p_test_all.append(p_test)
        p_train_all.append(p_train)
        r_test_all.append(r_test)
        r_train_all.append(r_train)

        print('Classifier Trained, run #{}'.format(i))
    mean_acc_test, std_acc_test = np.mean(np.array(accur_test)), np.std(np.array(accur_test))
    mean_acc_train, std_acc_train = np.mean(np.array(accur_train)), np.std(np.array(accur_train))
    mean_auroc_test, std_auroc_test = np.mean(np.array(auroc_test_all)), np.std(np.array(auroc_test_all))
    mean_auroc_train, std_auroc_train = np.mean(np.array(auroc_train_all)), np.std(np.array(auroc_train_all))
    mean_p_test, std_p_test = np.mean(np.array(p_test_all)), np.std(np.array(p_test_all))
    mean_r_test, std_r_test = np.mean(np.array(r_test_all)), np.std(np.array(r_test_all))
    mean_p_train, std_p_train = np.mean(np.array(p_train_all)), np.std(np.array(p_train_all))
    mean_r_train, std_r_train = np.mean(np.array(r_train_all)), np.std(np.array(r_train_all))
    print('Test accuracy = {0}+-{1}'.format(mean_acc_test, std_acc_test))
    print('Test AUROC = {0}+-{1}'.format(mean_auroc_test, std_auroc_test))
    print('Test Precision = {0}+-{1}'.format(mean_p_test, std_p_test))
    print('Test Recall = {0}+-{1}'.format(mean_r_test, std_r_test))
    print('Train accuracy = {0}+-{1}'.format(mean_acc_train, std_acc_train))
    print('Train AUROC = {0}+-{1}'.format(mean_auroc_train, std_auroc_train))
    print('Train Precision = {0}+-{1}'.format(mean_p_train, std_p_train))
    print('Train Recall = {0}+-{1}'.format(mean_r_train, std_r_train))


def run_classifier(x, y, centers, split=True):
    """
    Runs the classifier and returns the important features in "ascending" order
    :param x: input data (i.e. image histograms) of size (#images, n_cluster)
    :param y: corresponding labels of size (#images, #conditions)
    :param centers: centroid of clusters (gaze sequences) of size (350, 54)
    :param split: whether to split train and test data or not
    :return: important centroids sorted in ascending order, the classifier, and the percentiles
    """
    multi_run(x, y, args.num_run, split=split)
    acc, feat_imp, classifier = classifier_model(x, y, args.n_estimators, args.max_depth, args.max_features,
                                                 feat_importance=True, split=split)
    imp_feat, imp_feat_idx = np.sort(feat_imp), np.argsort(feat_imp)
    imp_centers = centers[imp_feat_idx]
    cumsum_feat = np.cumsum(np.flip(imp_feat, 0))
    percentile = (np.argmax(cumsum_feat > 0.5), np.argmax(cumsum_feat > 0.7), np.argmax(cumsum_feat > 0.9))
    return imp_centers, classifier, percentile


def train_test_split(x, y):
    num_imgs = y.shape[0]
    if num_imgs == 264:
        num_train = 220
    else:
        num_train = int(np.floor(0.8 * num_imgs))
    data = np.concatenate((x, y.reshape(num_imgs, -1)), axis=1)
    np.random.shuffle(data)
    train_x, train_y = data[:num_train, :args.n_cluster], data[:num_train, args.n_cluster:]
    test_x, test_y = data[num_train:, :args.n_cluster], data[num_train:, args.n_cluster:]
    return train_x, train_y, test_x, test_y


def plot_precision_recall(y, y_prob):
    Precision1, Recall1, thresholds = precision_recall_curve(y, y_prob[:, 0])
    Precision2, Recall2, thresholds = precision_recall_curve(y, y_prob[:, 1])
    fig = plt.figure()
    fig.set_size_inches(12, 12)
    ax1 = fig.add_subplot(1, 1, 1)
    plt.plot(Recall1, Precision1, lw=2, label='1st')
    plt.plot(Recall2, Precision2, lw=2, label='2nd')
    ax1.set_xlabel('Recall', size=18)
    ax1.set_ylabel('Precision', size=18)
    ax1.tick_params(labelsize=18)
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), fancybox=True, shadow=True)
    fig.savefig(args.path_to_videos + '/pr.png')


def precision_recall(y_true, y_pred):
    """
    Computes the precision and recall values for the positive class
    :param y_true: true labels
    :param y_pred: predicted labels
    """
    TP = FP = FN = TN = 0
    epsilon = 1e-4
    for i in range(len(y_pred)):
        if y_true[i] == 1 and y_pred[i] == 1:
            TP += 1
        elif y_true[i] == 0 and y_pred[i] == 1:
            FP += 1
        elif y_true[i] == 1 and y_pred[i] == 0:
            FN += 1
        elif y_true[i] == 0 and y_pred[i] == 0:
            TN += 1
    precision = (TP * 100.0) / (TP + FP + epsilon)
    recall = (TP * 100.0) / (TP + FN + epsilon)
    # return precision, recall, TP, TN, FP, FN
    return precision, recall
