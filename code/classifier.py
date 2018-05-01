import random
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from config import args


def classifier_model(x_input, y_input, n_estimators, max_depth, max_features, feat_importance=False):
    classifier = RandomForestClassifier(n_estimators=n_estimators,
                                        max_depth=max_depth,
                                        oob_score=True,
                                        max_features=max_features)
    x_train, y_train, x_test, y_test = train_test_split(x_input, y_input)
    classifier.fit(x_train, y_train)
    result = classifier.predict(x_test)
    # accuracy = np.sum(np.equal(np.argmax(result, 1), np.argmax(y_test, 1)))/np.float(y_test.shape[0])
    accuracy = np.sum(np.equal(np.reshape(result, (-1, 1)),
                               np.reshape(y_test, (-1, 1)))) / np.float(result.size)
    if feat_importance:
        features_imp = classifier.feature_importances_
        print('Classifier Trained & Feature importance generated')
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
    all_idx = range(len(x))
    train_idx = random.sample(all_idx, 220)
    test_idx = [item for item in all_idx if item not in train_idx]
    train_x = np.array(x)[train_idx]
    test_x = np.array(x)[test_idx]
    train_y = np.array(y)[train_idx]
    test_y = np.array(y)[test_idx]
    return train_x, train_y, test_x, test_y
