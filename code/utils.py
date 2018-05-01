import time
# import win32api
from scipy import spatial
import os
import operator
import numpy as np
import csv
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.ensemble import RandomForestClassifier
from config import args


def setCursor(data):
    for point in data:
        # win32api.SetCursorPos((int(point[0]), int(point[1])))
        time.sleep(0.1)


def dist(point1, point2):
    return ((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2) ** 0.5


def gazeExtractor(imgNames, gazePath):
    gaze_dict = {}
    for img in imgNames:
        gaze = []
        with open(gazePath + img, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            reader.next()
            for row in reader:
                gaze.append([int(row[0]), int(row[1])])
        gaze_dict[img[:-4].upper()] = gaze
    return gaze_dict


def gazeFilter(gaze_dict, distance, check_distance):
    modifiedGazeDict = dict(gaze_dict)
    for imgName in gaze_dict:
        gazeNew = gaze_dict[imgName]
        gaze2 = []
        gaze2.append(gazeNew[0])
        for i in range(len(gazeNew) - 1):
            if check_distance:
                if dist(gazeNew[i], gazeNew[i + 1]) > distance:
                    gaze_point = gazeNew[i + 1]
                    if gaze_point[0] <= 2560 and gaze_point[1] <= 1440:
                        gaze2.append(gazeNew[i + 1])
            else:
                gaze_point = gazeNew[i]
                if gaze_point[0] <= 2560 and gaze_point[1] <= 1440:
                    gaze2.append(gazeNew[i])
        modifiedGazeDict[imgName] = gaze2
    return modifiedGazeDict


def gazePlotter(data, name=None, flattened=True, save=False, output_path=None, plot=True):
    x = []
    y = []
    if flattened:
        for i in range(0, len(data) - 1, 2):
            x.append(data[i])
            y.append(data[i + 1])
    else:
        for point in data:
            x.append(point[0])
            y.append(point[1])
    fig = plt.figure()
    # plt.rcParams['animation.ffmpeg_path'] = 'C:/ffmpeg/ffmpeg.exe'
    plt.xlim(0, 2560)
    plt.ylim(1440, 0)
    line, = plt.plot(x, y)

    def animate(ii):
        line.set_data(x[:ii + 1], y[:ii + 1])
        return line

    writer = animation.ImageMagickFileWriter()  # for ubuntu
    # writers = animation.writers['ffmpeg']
    # writer = writers(fps=30, metadata=dict(artist='Me'), bitrate=1800)
    # writer = animation.FFMpegFileWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    img = plt.imread(args.proj_dir + '/images/CXR129_IM-0189-1001.jpg')
    plt.imshow(img)
    ani = animation.FuncAnimation(fig, animate, frames=50, interval=100)
    if save is True:
        if not os.path.exists(output_path + str(name.split('/')[0])):
            os.makedirs(output_path + str(name.split('/')[0]))
        ani.save(output_path + str(name) + '.mp4', writer='ffmpeg', fps=15)
    if plot is True:
        plt.show()


def rawSubPlotter(filteredGaze, imgName, output_path, save=False, plot=True, name=None):
    gaze_ = filteredGaze[imgName]
    gazePlotter(gaze_, flattened=False, save=save, output_path=output_path, plot=plot, name=name)


def gazeCluster(gazeSub, n_clust, init_centers, n_iter, n_init, calc_centres=False):
    print('go for clustering')
    clusterLabels = KMeans(n_clusters=n_clust, init=init_centers, random_state=0,
                           max_iter=n_iter, n_init=n_init).fit_predict(gazeSub)
    gazeSub = np.array(gazeSub)
    result = {i: gazeSub[np.where(clusterLabels == i)] for i in range(n_clust)}
    print('Completed Clustering')
    if calc_centres is True:
        centres = []
        for cluster in result:
            cluster_centre = np.mean(np.array(result[cluster]), axis=0)
            centres.append(cluster_centre)
        print ('Computed Centers')
        return np.array(centres), result
    return result


def input_generator(gaze_dict, subseq_len, n_cluster, csv_path, predict):
    gazeSub, gaze_sub_dict, carol_subs, darshan_subs, dians_subs = [], {}, [], [], []
    for img in gaze_dict:
        gaze = gaze_dict[img]
        img_sub_seqs = []
        for i in range(0, (len(gaze) - subseq_len), 2):
            flatArray = np.array(gaze[i:i + subseq_len]).flatten()
            gazeSub.append(flatArray)
            img_sub_seqs.append(flatArray)
            if img[:5].upper() == 'CAROL':
                carol_subs.append(flatArray)
            elif img[:7].upper() == 'DARSHAN':
                darshan_subs.append(flatArray)
            elif img[:5].upper() == 'DIANA':
                dians_subs.append(flatArray)
        gaze_sub_dict[img] = img_sub_seqs
    print('Extracted Sub-sequences')
    print ('Number of subs = {0}'.format(len(gazeSub)))
    centres, result = gazeCluster(gazeSub, n_cluster, 'k-means++', n_iter=300, n_init=10, calc_centres=True)
    print('Done Clustering')

    x, y = np.zeros((0, args.n_cluster)), np.zeros((0, 15))
    tree = spatial.KDTree(centres)
    with open(csv_path, 'r') as p:
        reader = csv.reader(p, delimiter='\t')
        reader.next()
        lines = [lin for lin in reader]
        img_name_list = [l[0].upper() for l in lines]
    image_names = []
    for img in gaze_sub_dict:
        row_number = img_name_list.index(img)
        row = lines[row_number]
        image_names.append(row[0])
        print(img)
        sub_seqs = gaze_sub_dict[img]
        x_ = np.zeros(shape=len(centres))
        for sub in sub_seqs:
            x_[tree.query(sub)[1]] += 1
        x = np.concatenate((x, np.array([i / np.sum(x_) for i in x_]).reshape(1, args.n_cluster)), axis=0)
        y = np.concatenate((y, np.array([int(row[i]) for i in range(1, 16)]).reshape(1, 15)), axis=0)
    return x, y, centres, carol_subs, darshan_subs, dians_subs, gaze_sub_dict, image_names
