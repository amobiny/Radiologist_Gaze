import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from scipy import spatial


def plot_important_features(img_dict, feat_dic, img_name, sorted_centers, sorted_feat_idx, num, n_clusters=100):
    """

    :param n_clusters: number of clusters used in KMeans clustering
    :param sorted_feat_idx:
    :param img_dict: dictionary containing all the info; including images, gaze data, histograms, etc.
    :param img_name: name of image to be plotted
    :param sorted_centers: important centers to be considered (sorted in ascending order of importance)
    """
    image = img_dict[img_name][1]
    gaze = img_dict[img_name][2]                # list of length #points including [x, y] coordinates
    hist = img_dict[img_name][0][:n_clusters] # n_cluster (100)
    feat = feat_dic[img_name][:, :416]        # #pointsx416
    # coords = feat_dic[image_name][:, 416:418]   # #gaze_pointsx2
    rad_label = img_dict[img_name][0][n_clusters:n_clusters+6]
    nih_label = img_dict[img_name][0][-6:]
    tree = spatial.KDTree(sorted_centers)
    cluster_lbl = np.array([])
    for i in range(feat.shape[0]):
        data_point = feat[i]
        cluster_lbl = np.append(cluster_lbl, int(tree.query(data_point)[1]))
    imp_features_idx = sorted_feat_idx[num]
    mask = np.isin(cluster_lbl, imp_features_idx)   # shows which points are important
    x = []
    y = []
    for point in gaze:
        x.append(point[0])
        y.append(point[1])

    fig, axs = plt.subplots(nrows=1, ncols=4)
    fig.set_size_inches(16, 6)

    ax = axs[0]
    ax.imshow(image, cmap='gray')
    ax.set_title(img_name)

    ax = axs[1]
    ax.imshow(image, cmap='gray')
    for i in range(len(mask)):
        if mask[i]:
            ax.plot(x[i], y[i], 'o', color='hotpink')
    ax.set_title(img_name)

    ax = axs[2]
    ax.plot(x, y)
    for i in range(len(mask)):
        if mask[i]:
            ax.plot(x[i], y[i], 'o', color='hotpink')
    ax.set_title(img_name)
    ax.imshow(image, cmap='gray')

    ax = axs[3]
    baar = ax.bar(range(len(hist)), hist)
    for imp in imp_features_idx:
        baar[imp].set_color('hotpink')
    # ax.bar(imp_features_idx, hist[imp_features_idx],)
    conds = ['Cardio. ', 'Infilt. ', 'Nodule ', 'Normal ', 'PlurEff. ', 'Pneumtrx.']
    cond_rad_lbl = np.where(rad_label == 1)[0]
    cond_nih_lbl = np.where(nih_label == 1)[0]
    tit = [conds[a] for a in cond_rad_lbl] + ['\n NIH:'] + [conds[a] for a in cond_nih_lbl]
    img_title = 'Radiologist: ' + '-'.join(tit)
    ax.set_title(img_title)
    plt.show()
    print()


def plot_box(img_dict, feat_dic, img_name, sorted_centers, sorted_feat_idx, num, n_clusters=100):
    """

    :param img_dict: dictionary containing all the info; including images, gaze data, histograms, etc.
    :param img_name: name of image to be plotted
    :param sorted_centers: important centers to be considered (sorted in ascending order of importance)
    """
    image = img_dict[img_name][1]
    gaze = img_dict[img_name][2]                # list of length #points including [x, y] coordinates
    hist = img_dict[img_name][0][:n_clusters] # n_cluster (100)
    feat = feat_dic[img_name][:, :416]        # #pointsx416
    # coords = feat_dic[image_name][:, 416:418]   # #gaze_pointsx2
    rad_label = img_dict[img_name][0][n_clusters:n_clusters+6]
    nih_label = img_dict[img_name][0][-6:]
    tree = spatial.KDTree(sorted_centers)
    cluster_lbl = np.array([])
    for i in range(feat.shape[0]):
        data_point = feat[i]
        cluster_lbl = np.append(cluster_lbl, int(tree.query(data_point)[1]))
    imp_features_idx = sorted_feat_idx[-num]
    mask = np.isin(cluster_lbl, imp_features_idx) # shows which points are important

    fig, axs = plt.subplots(nrows=1, ncols=2)
    fig.set_size_inches(12, 6)
    ax = axs[0]
    x = []
    y = []
    for point in gaze:
        x.append(point[0])
        y.append(point[1])
    # ax.plot(x, y)
    # ax.plot(x[-1], y[-1], 'o', color='red', markersize=8)
    # ax.plot(x[0], y[0], 'o', color='lawngreen', markersize=8)
    for i in range(len(mask)):
        if mask[i]:
            rect = patches.Rectangle((x[i]-15, y[i]-15), 30, 30, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            # ax.plot(x[i], y[i], 'o', color='hotpink')
    tit = img_name + ', important feature #' + str(num)
    ax.set_title(tit)
    ax.imshow(image, cmap='gray')
    ax = axs[1]
    baar = ax.bar(range(len(hist)), hist)
    baar[int(imp_features_idx)].set_color('r')
    # ax.bar(imp_features_idx, hist[imp_features_idx],)
    conds = ['Cardio. ', 'Infilt. ', 'Nodule ', 'Normal ', 'PlurEff. ', 'Pneumtrx.']
    cond_rad_lbl = np.where(rad_label == 1)[0]
    cond_nih_lbl = np.where(nih_label == 1)[0]
    tit = [conds[a] for a in cond_rad_lbl] + ['\n NIH:'] + [conds[a] for a in cond_nih_lbl]
    img_title = 'Radiologist: ' + '-'.join(tit)
    ax.set_title(img_title)
    plt.show()
    print()
