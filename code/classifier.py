import random
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve
from config import args
import matplotlib.pyplot as plt


def classifier_model(x_input, y_input, n_estimators, max_depth, max_features, feat_importance=False):
    classifier = RandomForestClassifier(n_estimators=n_estimators,
                                        max_depth=max_depth,
                                        oob_score=True,
                                        max_features=max_features)
    x_train, y_train, x_test, y_test = train_test_split(x_input, y_input)
    classifier.fit(x_train, y_train)
    result = classifier.predict(x_test)
    y_prob_test = classifier.predict_proba(x_test)
    # accuracy = np.sum(np.equal(np.argmax(result, 1), np.argmax(y_test, 1)))/np.float(y_test.shape[0])
    accuracy = np.sum(np.equal(np.reshape(result, (-1, 1)),
                               np.reshape(y_test, (-1, 1)))) / np.float(result.size)
    if feat_importance:
        features_imp = classifier.feature_importances_
        print('Classifier Trained & Feature importance generated')
        plot_precision_recall(y_test, y_prob_test)
        return accuracy, features_imp
    else:
        return accuracy


def multi_run(x, y, count):
    accur = []
    for i in range(count):
        acc = classifier_model(x, y, args.n_estimators, args.max_depth, args.max_features)
        accur.append(acc)
        print('Classifier Trained, run #{}'.format(i))
    mean_acc = np.mean(np.array(accur))
    std_acc = np.std(np.array(accur))
    return mean_acc, std_acc


def run_classifier(x, y, centers):
    mean_acc, std_acc = multi_run(x, y, args.num_run)
    print('Average accuracy over {0} runs: {1:.02%}+-({2:.2f})'.format(args.num_run, mean_acc, std_acc*100))
    acc, feat_imp = classifier_model(x, y, args.n_estimators, args.max_depth, args.max_features, feat_importance=True)
    imp_feat, imp_feat_idx = np.sort(feat_imp), np.argsort(feat_imp)
    imp_centers = centers[imp_feat_idx]
    return imp_centers


def train_test_split(x, y):
    data = np.concatenate((x, y.reshape(264, -1)), axis=1)
    np.random.shuffle(data)
    train_x, train_y = data[:220, :args.n_cluster], data[:220, args.n_cluster:]
    test_x, test_y = data[220:, :args.n_cluster], data[220:, args.n_cluster:]
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
