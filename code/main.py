from utils import *
from gazeWarp import *
import operator
import cPickle as pickle
from config import args
from classifier import *

gaze_file_names = os.listdir(args.all_gaze_path)
raw_gaze_dict = gazeExtractor(gaze_file_names, args.all_gaze_path)
xml_files = os.listdir(args.xml_path)


def get_cluster_count(imp_centers, subs):
    """
    counts the number of sub-sequences in 'subs' which belongs to each cluster
    :param imp_centers: numpy array of size [n_cluster, subseq_len*2]   [350, 54]
                        first dimension is sorted based on the feature importance (in ascending order)
    :param subs: list of all sub-sequences; each sub-sequence is an array of size [subseq_len*2, ]
    :return:
    """
    center_tree = spatial.KDTree(imp_centers)
    cluster_count = {i: 0 for i in range(len(imp_centers))}
    for sub in subs:
        cluster_count[center_tree.query(sub)[1]] += 1
    cluster_count = cluster_count.items()
    # cluster_sorted = sorted(cluster_count.items(), key=operator.itemgetter(1), reverse=True)
    return cluster_count


def gaze_plot_save(cluster_sorted, imp_centers, num=5, path='carol_most_important/'):
    for i in range(num):
        gazePlotter(imp_centers[cluster_sorted[i][0]], name=path + str(i), flattened=True,
                    save=True, output_path=args.path_to_videos, plot=False)
        print('Creating and Saving Animation.....')


if __name__ == "__main__":
    # rawSubPlotter(normal_gaze, img_name, save=True, output_path=out_path, plot=False, name='Normal')
    if args.load_warped_gaze:
        with open('warped_gaze.pkl', 'rb') as inputs:
            filtered_gaze_dict, xml_dict, warped_dict = pickle.load(inputs)
            print('Warped Gaze Loaded')
    else:
        filtered_gaze_dict = gazeFilter(raw_gaze_dict, distance=args.filter_distance, check_distance=True)
        print('Done Filtering')
        xml_dict = dict_creator(args.xml_path, xml_files)
        print('Annotations Extracted')
        warped_dict = warping(xml_dict, filtered_gaze_dict, args.reference_image, gaze_path=None, save=False)
        print('Warping complete')

    if args.prepare_data:
        x, y, centers, carol_subs, darshan_subs, diana_subs, img_sub_dict, image_names = input_generator(warped_dict,
                                                                                            subseq_len=args.subseq_len,
                                                                                            n_cluster=args.n_cluster,
                                                                                            csv_path=args.label_csv,
                                                                                            predict='error')
        with open('Data_nodule_error.pkl', 'wb') as output:
            pickle.dump([x, y, centers, carol_subs, darshan_subs, diana_subs, img_sub_dict, image_names], output, -1)
        print('Input Generated')
    else:
        with open('Data.pkl', 'rb') as inputs:
            x, y, centers, carol_subs, darshan_subs, diana_subs, img_sub_dict, image_names = pickle.load(inputs)
        print('Input Loaded')

    sorted_cluster_center_imp = run_classifier(x, y[:, 7:10], centers)  # returns array of size [350, 54]

    if args.load_img_sub_count:
        with open('img_sub_count.pkl', 'rb') as inputs:
            img_sub_count_dict = pickle.load(inputs)
    else:
        print('Preparing the dictionary of sub-sequence counts for each image-radiologist pair')
        img_sub_count_dict = {}
        for key in img_sub_dict.keys():
            print(key)
            img_sub_count_dict[key] = get_cluster_count(sorted_cluster_center_imp, img_sub_dict[key])
        with open('img_sub_count.pkl', 'wb') as output:
            pickle.dump(img_sub_count_dict, output, -1)

    all_subs = carol_subs + darshan_subs + diana_subs
    sorted_all_cluster_count = get_cluster_count(sorted_cluster_center_imp, all_subs)  # list of 350 tuples
    gaze_plot_save(sorted_all_cluster_count, sorted_cluster_center_imp, num=args.vidnum, path='all_most_important/')
