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
        pickle.dump([x, y, centers, carol_subs, darshan_subs, diana_subs], output, -1)
    print('Input Generated and Saved')
else:
    with open('Data.pkl', 'rb') as inputs:
        x, y, centers, carol_subs, darshan_subs, diana_subs, img_sub_dict, image_names = pickle.load(inputs)
    print('Input Loaded')

# 0.Cardio, 1.Infilt, 2.Nodule, 3. Normal, 4.PlurEff, 5.Pneumtrx, 6.Carol, 7.Darshan, 8.Diana
# 9.Cardio, 10.Infilt, 11.Nodule, 12. Normal, 13.PlurEff, 14.Pneumtrx,
# sorted_cluster_center_imp = run_classifier(x, np.reshape(np.abs(y[:, 0]-y[:, 0+9]), (264, -1)), centers)
sorted_cluster_center_imp = run_classifier(x, y[:, 6:9], centers)
# returns array of size (350, 54)

carol_cluster_count = get_cluster_count(sorted_cluster_center_imp, carol_subs)  # list of 350 tuples
gaze_plot_save(carol_cluster_count, sorted_cluster_center_imp, num=args.numvid, path='radiologist_most_important/')
darshan_cluster_count = get_cluster_count(sorted_cluster_center_imp, darshan_subs)  # list of 350 tuples
# gaze_plot_save(darshan_cluster_count, sorted_cluster_center_imp, num=args.numvid, path='darshan_nodule_most_important/')
diana_cluster_count = get_cluster_count(sorted_cluster_center_imp, diana_subs)  # list of 350 tuples
# gaze_plot_save(diana_cluster_count, sorted_cluster_center_imp, num=args.numvid, path='diana_nodule_most_important/')

print()
# if args.load_img_sub_count:
#     with open('img_sub_count.pkl', 'rb') as inputs:
#         img_sub_count_dict = pickle.load(inputs)
# else:
#     print('Preparing the dictionary of sub-sequence counts for each image-radiologist pair')
#     img_sub_count_dict = {}
#     for key in img_sub_dict.keys():
#         print(key)
#         img_sub_count_dict[key] = get_cluster_count(sorted_cluster_center_imp, img_sub_dict[key])
#     with open('img_sub_count.pkl', 'wb') as output:
#         pickle.dump(img_sub_count_dict, output, -1)
#
# all_subs = carol_subs + darshan_subs + diana_subs
# sorted_all_cluster_count = get_cluster_count(sorted_cluster_center_imp, all_subs)  # list of 350 tuples
# gaze_plot_save(sorted_all_cluster_count, sorted_cluster_center_imp, num=args.vidnum, path='all_most_important/')

print('**********************Carol****************************')
print([i for j, i in carol_cluster_count][-10:])
print('Sum:{}'.format(np.sum([i for j, i in carol_cluster_count][-10:])))
print('Per image: {}'.format(np.sum([i/102. for j, i in carol_cluster_count][-10:])))
print('Percentage: {}'.format(np.sum([i/22028.*100 for j, i in carol_cluster_count][-10:])))

print('**********************Darshan****************************')
print([i for j, i in darshan_cluster_count][-10:])
print('Sum:{}'.format(np.sum([i for j, i in darshan_cluster_count][-10:])))
print('Per image: {}'.format(np.sum([i/60. for j, i in darshan_cluster_count][-10:])))
print('Percentage: {}'.format(np.sum([i/9261.*100 for j, i in darshan_cluster_count][-10:])))


print('**********************Diana****************************')
print([i for j, i in darshan_cluster_count][-10:])
print('Sum:{}'.format(np.sum([i for j, i in diana_cluster_count][-10:])))
print('Per image: {}'.format(np.sum([i/102. for j, i in diana_cluster_count][-10:])))
print('Percentage: {}'.format(np.sum([i/39966.*100 for j, i in diana_cluster_count][-10:])))
