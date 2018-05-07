import random
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve
from config import args
import matplotlib.pyplot as plt


def classifier_model(x_input, y_input, n_estimators, max_depth, max_features, feat_importance=False, split=True):
    classifier = RandomForestClassifier(n_estimators=n_estimators,
                                        max_depth=max_depth,
                                        oob_score=True,
                                        max_features=max_features)
    if split:
        x_train, y_train, x_test, y_test = train_test_split(x_input, y_input)
    else:
        x_test = x_train = x_input
        y_test = y_train = y_input
    classifier.fit(x_train, y_train)
    result = classifier.predict(x_test)
    y_prob_test = classifier.predict_proba(x_test)
    # accuracy = np.sum(np.equal(np.argmax(result, 1), np.argmax(y_test, 1)))/np.float(y_test.shape[0])
    accuracy = np.sum(np.equal(np.reshape(result, (-1, 1)),
                               np.reshape(y_test, (-1, 1)))) / np.float(result.size)
    if feat_importance:
        features_imp = classifier.feature_importances_
        print('Classifier Trained & Feature importance generated')
        # plot_precision_recall(y_test, y_prob_test)
        return accuracy, features_imp, classifier
    else:
        return accuracy


def multi_run(x, y, count, split=True):
    accur = []
    for i in range(count):
        acc = classifier_model(x, y, args.n_estimators, args.max_depth, args.max_features, split=split)
        accur.append(acc)
        print('Classifier Trained, run #{}'.format(i))
    mean_acc = np.mean(np.array(accur))
    std_acc = np.std(np.array(accur))
    return mean_acc, std_acc


def run_classifier(x, y, centers, split=True):
    mean_acc, std_acc = multi_run(x, y, args.num_run, split=split)
    print('Average accuracy over {0} runs: {1:.02%}+-({2:.2f})'.format(args.num_run, mean_acc, std_acc * 100))
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
