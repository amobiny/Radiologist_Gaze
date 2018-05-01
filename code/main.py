from utils_animplot import gaze_plot_save
from utils import *
from gazeWarp import *
import operator
import cPickle as pickle
from config import args
from classifier import *

gaze_file_names = os.listdir(args.all_gaze_path)
raw_gaze_dict = gazeExtractor(gaze_file_names, args.all_gaze_path)
xml_files = os.listdir(args.xml_path)


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
                                                                                        csv_path=args.label_csv)
    with open('Data.pkl', 'wb') as output:
        pickle.dump([x, y, centers, carol_subs, darshan_subs, diana_subs, img_sub_dict, image_names], output, -1)
    print('Input Generated and Saved')
else:
    with open('Data.pkl', 'rb') as inputs:
        x, y, centers, carol_subs, darshan_subs, diana_subs, img_sub_dict, image_names = pickle.load(inputs)
    print('Input Loaded')

# 1.Cardio, 2.Infilt, 3.Nodule, 4. Normal, 5.PlurEff, 6.Pneumtrx, 7.Carol, 8.Darshan, 9.Diana
# 10.Cardio, 11.Infilt, 12.Nodule, 13. Normal, 14.PlurEff, 15.Pneumtrx,
# sorted_cluster_center_imp = run_classifier(x, np.reshape(np.abs(y[:, 1]-y[:, 1+9]), (264, -1)), centers)
sorted_cluster_center_imp = run_classifier(x, y[:, 7:10], centers)
# returns array of size (350, 54)

carol_cluster_count = get_cluster_count(sorted_cluster_center_imp, carol_subs)  # list of 350 tuples
gaze_plot_save(carol_cluster_count, sorted_cluster_center_imp, num=args.numvid, path='carol_most_important/')
darshan_cluster_count = get_cluster_count(sorted_cluster_center_imp, darshan_subs)  # list of 350 tuples
gaze_plot_save(darshan_cluster_count, sorted_cluster_center_imp, num=args.numvid, path='darshan_most_important/')
diana_cluster_count = get_cluster_count(sorted_cluster_center_imp, diana_subs)  # list of 350 tuples
gaze_plot_save(diana_cluster_count, sorted_cluster_center_imp, num=args.numvid, path='diana_most_important/')

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
