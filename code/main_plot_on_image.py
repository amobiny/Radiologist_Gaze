from classifier import run_classifier
import cPickle as pickle
from scipy import spatial
import numpy as np

from utils_animplot import *

with open('Data.pkl', 'rb') as inputs:
    x, y, centers, carol_subs, darshan_subs, diana_subs, img_sub_dict, image_names = pickle.load(inputs)

sorted_cluster_center_imp, _, _ = run_classifier(x, y[:, 2], centers, split=True)


def subseq_for_image(cluster_center, img_sub_dict, radiologist_name):
    center_tree = spatial.KDTree(cluster_center)
    rad_dict = {}
    for img_name, subs in img_sub_dict.items():
        # loop over images
        if img_name.split('_')[0].upper() == radiologist_name:
            print(img_name)
            image_name = img_name.split(radiologist_name)[1][1:]
            rad_dict[image_name] = {i: np.zeros((0, 54)) for i in range(len(cluster_center))}
            for sub in subs:
                # loop over the sub-sequences of the selected image to put them inside the corresponding clusters
                cluster_num = center_tree.query(sub)[1]
                rad_dict[image_name][cluster_num] = np.concatenate((rad_dict[image_name][cluster_num], sub.reshape(1, -1)), axis=0)
    return rad_dict


darshan_dict = subseq_for_image(sorted_cluster_center_imp, img_sub_dict, radiologist_name='DARSHAN')
assert (np.sum([a.shape[0] for a in darshan_dict['CXR1055_IM-0040-1001'].values()]) ==
        len(img_sub_dict['DARSHAN_CXR1055_IM-0040-1001'])), "Error Detected"

path = 'test_path/'
data = list(darshan_dict['CXR1055_IM-0040-1001'][349][0])
gaze_img_Plotter(data, name=path+'test', save=True, output_path=args.path_to_videos, img_name='CXR1055_IM-0040-1001')

def gaze_img_Plotter(all_data, name=None, flattened=True, save=False, output_path=None, plot=True, img_name=None):
    col = ['dodgerblue', 'mediumslateblue', 'orangered', 'brown']
    if len(all_data) == 0:
        img = plt.imread(args.proj_dir + '/images/' + img_name + '.jpg')
        plt.imshow(img)
    else:
        for i in range(len(all_data)):
            data = all_data[i]
            x = []
            y = []
            if not img_name:
                img_name = 'CXR129_IM-0189-1001'
            if flattened:
                for i in range(0, len(data) - 1, 2):
                    x.append(data[i])
                    y.append(data[i + 1])
            else:
                for point in data:
                    x.append(point[0])
                    y.append(point[1])
            fig = plt.figure()
            plt.xlim(0, 2560)
            plt.ylim(1440, 0)
            plt.plot(x, y)
            plt.plot(x[-1], y[-1], 'o')
            img = plt.imread(args.proj_dir + '/images/' + img_name + '.jpg')
            plt.imshow(img)

print()
