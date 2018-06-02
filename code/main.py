from utils_animplot import gaze_plot_save
from utils import *
from gazeWarp import *
import operator
import cPickle as pickle
from config import args
from classifier import *
import matplotlib.patches as patches
from scipy.stats import chisquare

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

    img_str_list = []
    img_name_list = []
    for key, value in xml_dict.iteritems():
        radiologist_name = key.split('_')[0].upper()
        img_name = key.split(radiologist_name)[1][1:]
        if img_name not in img_name_list:
            a = sum(value, [])
            b = [str(int(x)) for x in a]
            b.insert(0, img_name)
            img_str = '*'.join(b)
            img_str_list.append(img_str)
            img_name_list.append(img_name)

    # thefile = open('test.txt', 'w')
    # for item in img_str_list:
    #     # thefile.write("%s\n" % item)
    #     print(item)
    #     thefile.write('{}\n'.format(item))

    # myFile = open('Annotations.csv', 'w')
    # with myFile:
    #     writer = csv.writer(myFile)
    #     writer.writerows(img_str_list)
    # myFile = open('image_name_list.csv', 'w')
    # with myFile:
    #     writer = csv.writer(myFile)
    #     writer.writerows(img_name_list)

    warped_dict = warping(xml_dict, filtered_gaze_dict, args.reference_image, gaze_path=None, save=False, image_path=args.path_to_images)
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
    # with open('Data.pkl', 'rb') as inputs:
    #     x, y, centers, carol_subs, darshan_subs, diana_subs, img_sub_dict, image_names = pickle.load(inputs)
    # print('Input Loaded')
    with open('Data.pkl', 'rb') as inputs:
        _, _, _, _, _, _, img_sub_dict, _ = pickle.load(inputs)
    print('Input Loaded')
# condition_num = 0

# 0.Cardio, 1.Infilt, 2.Nodule, 3.Normal, 4.PlurEff, 5.Pneumtrx, 6.Carol, 7.Darshan, 8.Diana
# 9.Cardio, 10.Infilt, 11.Nodule, 12. Normal, 13.PlurEff, 14.Pneumtrx,
# sorted_cluster_center_imp = run_classifier(x, np.reshape(np.abs(y[:, condition_num]-y[:, condition_num+9]), (264, -1)), centers)

# x_new, y_new = x, y
# x_new, y_new = pick_radiologist(x, y, image_names, radiologist_name='DARSHAN')
# sorted_cluster_center_imp, classifier, percentile = run_classifier(x_new, y_new[:, 2], centers, split=True)
# returns array of size (350, 54)

# carol_cluster_count = get_cluster_count(sorted_cluster_center_imp, carol_subs)  # list of 350 tuples

def gaze_img_Plotter(data, image):
    x = []
    y = []
    for i in range(0, len(data) - 1, 2):
        x.append(data[i])
        y.append(data[i + 1])
    plt.plot(x, y, linewidth=2, color='yellow')
    # plt.plot(x[-1], y[-1], 'o')
    plt.imshow(image, cmap='gray')
    plt.savefig(out_dir + '0_ref_image' + '.png')
    # plt.show()


def cluster_num_finder(sorted_center, img_dict):
    """
    finds the cluster number for all sub-sequences of all images
    :param sorted_center: cluster centers sorted in ascending order
    :param img_dict: dictionary of images and associated sub-sequences
    :return: dictionary of: image_name:label of all sub-sequences
    """
    cluster_num_dict = {}
    tree = spatial.KDTree(sorted_center)
    for img, sub_seqs in img_dict.items():
        print(img)
        cluster_num_dict[img] = np.zeros((len(sub_seqs)))
        for ii in range(len(sub_seqs)):
            sub = sub_seqs[ii]
            cluster_num_dict[img][ii] = tree.query(sub)[1]
    return cluster_num_dict


# seq_cluster_label_dict = cluster_num_finder(sorted_cluster_center_imp, img_sub_dict)

# with open('cluster_num_dict.pkl', 'wb') as output:
#     pickle.dump([sorted_cluster_center_imp, seq_cluster_label_dict], output, -1)

with open('cluster_num_dict.pkl', 'rb') as inputs:
    sorted_cluster_center_imp, seq_cluster_label_dict = pickle.load(inputs)

# adding labels to the dictionary
with open(args.label_csv, 'r') as p:
    reader = csv.reader(p, delimiter='\t')
    reader.next()
    lines = [lin for lin in reader]
    img_name_list = [l[0].upper() for l in lines]

img_label_dict = {}
nodule_label_dict = {}
feature_count_dict = {}
for i in range(len(img_name_list)):
    image_name = img_name_list[i]
    img_label_dict[image_name] = np.array([int(lines[i][j]) for j in range(1, 16)])
    nodule_label_dict[image_name] = np.array([img_label_dict[image_name][2], img_label_dict[image_name][11]])
    if image_name in seq_cluster_label_dict.keys():
        feature_count_dict[image_name] = np.zeros(args.n_cluster)
        unqs = np.unique(seq_cluster_label_dict[image_name], return_counts=True)
        for j in range(len(unqs[0])):
            feature_count_dict[image_name][int(unqs[0][j])] = unqs[1][j]


def hist_generator(counts, n_bins):
    histt = np.zeros(n_bins)
    unqs = np.unique(counts, return_counts=True)
    for i in range(len(unqs[0])):
        histt[int(unqs[0][i])] = unqs[1][i]
    return histt


# pvals = np.zeros(args.n_cluster)
# counts = np.zeros((args.n_cluster, 4))
# counts_percentage = np.zeros((args.n_cluster, 4))
# for which_feature in range(args.n_cluster):
#     correct_count, incorrect_count = np.array([]), np.array([])
#     for image_name in feature_count_dict.keys():
#         if nodule_label_dict[image_name][0] == nodule_label_dict[image_name][1]:
#             correct_count = np.append(correct_count, feature_count_dict[image_name][which_feature])
#         else:
#             incorrect_count = np.append(incorrect_count, feature_count_dict[image_name][which_feature])
#
#     a = np.sum(correct_count == 0)
#     b = np.sum(correct_count > 0)
#     aa = np.sum(incorrect_count == 0)
#     bb = np.sum(incorrect_count > 0)
#     _, pvals[which_feature] = chisquare([a*100.0 / correct_count.shape, b*100.0 / correct_count.shape],
#                                         f_exp=[aa*100.0 / incorrect_count.shape, bb*100.0 / incorrect_count.shape])
#     counts[which_feature, :] = np.array([a, b, aa, bb])
#     counts_percentage[which_feature, :] = np.array([a*100.0 / correct_count.shape, b*100.0 / correct_count.shape,
#                                                     aa*100.0 / incorrect_count.shape, bb*100.0 / incorrect_count.shape]).reshape(1,4)

# sig_feat_idx = np.where(pvals < 0.001)[0]
# ref_image = plt.imread(args.proj_dir + '/images/' + 'CXR129_IM-0189-1001' + '.jpg')
#
# for feat_idx in sig_feat_idx:
#     fig, axs = plt.subplots(nrows=1, ncols=2)
#     fig.set_size_inches(12, 6)
#     gaze = sorted_cluster_center_imp[feat_idx]
#     x = []
#     y = []
#     for i in range(0, len(gaze) - 1, 2):
#         x.append(gaze[i])
#         y.append(gaze[i + 1])
#     axs[0].plot(x, y)
#     axs[0].imshow(ref_image, cmap='gray')
#     axs[0].set_title('feature # {0}, p_value={1:.6f}'.format(feat_idx, pvals[feat_idx]))
#     axs[1].bar(range(4), counts[feat_idx])
#     plt.xticks(range(4), ['cor/wo', 'cor/w', 'inc/wo', 'inc/w'], rotation='vertical')
#     plt.ylim([0, 250])
#     for i in range(4):
#         plt.text(i, 40, str(int(counts[feat_idx][i])))
#         plt.text(i, 20, '(' + str(int(counts_percentage[feat_idx][i])) + '%)')
#     plt.savefig('/home/cougarnet.uh.edu/amobiny/Desktop/Radiologist_Gaze/significant_features/feature_{}.png'.format(feat_idx))


def plot_on_original_image(img_name, f_num, img_dict, label_dict, output_dir, x_l):
    radiologist_name = img_name.split('_')[0].upper()
    img_n = img_name.split(radiologist_name)[1][1:]
    image = plt.imread(args.proj_dir + '/warped_images/' + img_n + '.jpg')
    subs = img_dict[img_name]
    labels = label_dict[img_name]
    sub_idx = list(np.where(labels == f_num)[0].astype(int))
    if not len(sub_idx):
        # plt.imshow(image, cmap='gray')
        print('no subsequence from feature_number={0} in image {1}'.format(f_num, img_name))
        # plt.title(img_name + ' has no subsecuence in feature #' + str(f_num))
    else:
        os.makedirs(output_dir+img_name)
        new_subs = [subs[a] for a in sub_idx]
        seq_num = 0
        for sub in new_subs:
            seq_num += 1
            x = []
            y = []
            for jj in range(0, len(sub) - 1, 2):
                x.append(sub[jj])
                y.append(sub[jj + 1])
            for fig_num in range(3):
                if fig_num == 0:
                    plt.figure()
                    plt.imshow(image, cmap='gray')
                    plt.plot(x, y, linewidth=2, color='yellow')
                    plt.title('{0} \n subsequence #{1} (out of {2}) of feature #{3}'.format(img_name, seq_num, len(sub_idx), f_num))
                    plt.xlabel(x_l)
                    plt.savefig(output_dir + img_name + '/' + str(seq_num) + '_' + str(fig_num) + '.png')
                elif fig_num == 1:
                    pass
                    # fig, ax = plt.subplots(nrows=1, ncols=1)
                    # plt.imshow(image, cmap='gray')
                    # x_min, x_max = np.min(np.array(x)), np.max(np.array(x))
                    # y_min, y_max = np.min(np.array(y)), np.max(np.array(y))
                    # rect = patches.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min, linewidth=2, edgecolor='r', facecolor='none')
                    # ax.add_patch(rect)
                    # plt.title('{0} \n subsequence #{1} (out of {2}) of feature #{3}'.format(img_name, seq_num, len(sub_idx), f_num))
                    # plt.xlabel(x_l)
                    # plt.savefig(output_dir + img_name + '/' + str(seq_num) + '_' + str(fig_num) + '.png')
                elif fig_num == 2:
                    pass
                    # crop_image = image[y_min:y_max, x_min:x_max]
                    # plt.imshow(crop_image, cmap='gray')
                    # plt.title('{0} \n subsequence #{1} (out of {2}) of feature #{3}'.format(img_name, seq_num, len(sub_idx), f_num))
                    # plt.xlabel(x_l)
                    # plt.savefig(output_dir + img_name + '/' + str(seq_num) + '_' + str(fig_num) + '.png')
        # plot all interpolated
        plt.figure()
        plt.imshow(image, cmap='gray')
        for sub in new_subs:
            x = []
            y = []
            for jj in range(0, len(sub) - 1, 2):
                x.append(sub[jj])
                y.append(sub[jj + 1])
            plt.plot(x, y)
        plt.title('{0} has {1} subsequence in feature #{2}'.format(img_name, len(sub_idx), f_num))
        plt.xlabel(x_l)
        plt.savefig(output_dir + img_name + '/' + '0_orig_image.png')


which_feature = 349
class_label = ['no-nodule', 'nodule']
out_dir = '/home/cougarnet.uh.edu/amobiny/Desktop/Radiologist_Gaze/code/feat_349_nodule/'
ref_image = plt.imread(args.proj_dir + '/images/' + 'CXR129_IM-0189-1001' + '.jpg')
gaze_img_Plotter(list(sorted_cluster_center_imp[which_feature, :]), ref_image)
for image_name in img_sub_dict.keys():
    x_label = 'NIH: ' + class_label[img_label_dict[image_name][11]] + \
              ', Radiologist: ' + class_label[img_label_dict[image_name][2]]
    plot_on_original_image(image_name, which_feature, img_sub_dict, seq_cluster_label_dict, out_dir, x_label)


######### Dive into features. Why getting these results?!
# let's look deep into them. Feature #248 and #302 looks interesting to look into.


def look_at_feature(feature_count_d, nodule_label_d, feat_num, out_dir):
    """
    finds images with a particular feature, the count of feature in each of those images, and the labels
    :param feature_count_d: dictionary of images and their corresponding feature counts
    :param nodule_label_d: dictionary of labels from NIH and radiologist provided for nodule
    :param feat_num: number of feature of concern
    :return: dictionary:: image_name:(count_of_feature, radiologist diagnosis, NIH label)
    """
    out_dict = {}
    rad_pred_dict = {'CAROL': np.zeros(4), 'DARSHAN': np.zeros(4), 'DIANA': np.zeros(4)}    # (TP, TN, FP, FN)
    class_lab = ['no-nodule', 'nodule']
    for image_name in feature_count_d.keys():
        if feature_count_d[image_name][feat_num]:
            out_dict[image_name] = (feature_count_d[image_name][feat_num],
                                    nodule_label_d[image_name][0],
                                    nodule_label_d[image_name][1])
            radiologist_name = image_name.split('_')[0].upper()
            if nodule_label_d[image_name][0] and nodule_label_d[image_name][1]:   # TP
                rad_pred_dict[radiologist_name][0] += 1
            elif not nodule_label_d[image_name][0] and not nodule_label_d[image_name][1]:  # TN
                rad_pred_dict[radiologist_name][1] += 1
            elif nodule_label_d[image_name][0] and not nodule_label_d[image_name][1]:  # FP
                rad_pred_dict[radiologist_name][2] += 1
            elif not nodule_label_d[image_name][0] and nodule_label_d[image_name][1]:  # FN
                rad_pred_dict[radiologist_name][3] += 1

            # x_label = 'NIH: ' + class_lab[nodule_label_d[image_name][1]] + \
            #           ', Radiologist: ' + class_lab[nodule_label_d[image_name][0]]
            # plot_on_original_image(image_name, feat_num, img_sub_dict, seq_cluster_label_dict, out_dir,
            #                        x_label)
    print('CAROL: #TP={0}, #TN={1}, #FP={2}, #FN={3}\n'.format(rad_pred_dict['CAROL'][0], rad_pred_dict['CAROL'][1], rad_pred_dict['CAROL'][2], rad_pred_dict['CAROL'][3]))
    print('DARSHAN: #TP={0}, #TN={1}, #FP={2}, #FN={3}\n'.format(rad_pred_dict['DARSHAN'][0], rad_pred_dict['DARSHAN'][1], rad_pred_dict['DARSHAN'][2], rad_pred_dict['DARSHAN'][3]))
    print('DIANA: #TP={0}, #TN={1}, #FP={2}, #FN={3}\n'.format(rad_pred_dict['DIANA'][0], rad_pred_dict['DIANA'][1], rad_pred_dict['DIANA'][2], rad_pred_dict['DIANA'][3]))

# out_dir = '/home/cougarnet.uh.edu/amobiny/Desktop/Radiologist_Gaze/code/feat_248_nodule/'
# look_at_feature(feature_count_dict, nodule_label_dict, 302, out_dir)
# print()








# save_histogram(carol_cluster_count, percentile, name='/darshan_all_abnorms_most_important2.png')
gaze_plot_save(carol_cluster_count, sorted_cluster_center_imp, num=args.numvid,
               path='darshan_all_abnorms_most_important2/')
# darshan_cluster_count = get_cluster_count(sorted_cluster_center_imp, darshan_subs)  # list of 350 tuples
# gaze_plot_save(darshan_cluster_count, sorted_cluster_center_imp, num=args.numvid, path='darshan_nodule_most_important/')
# diana_cluster_count = get_cluster_count(sorted_cluster_center_imp, diana_subs)  # list of 350 tuples
# gaze_plot_save(diana_cluster_count, sorted_cluster_center_imp, num=args.numvid, path='diana_nodule_most_important/')


# a1, b1 = pick_radiologist(x, y, image_names, radiologist_name='CAROL')
# c1 = classifier.score(a1, b1[:, condition_num])
# print('accuracy: {0:.02%}'.format(c1))
# a2, b2 = pick_radiologist(x, y, image_names, radiologist_name='DARSHAN')
# c2 = classifier.score(a2, b2[:, condition_num])
# print('accuracy: {0:.02%}'.format(c2))
# print()
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
print([i / 102. for j, i in carol_cluster_count][-10:])
print([i / 22028. * 100 for j, i in carol_cluster_count][-10:])
print('Sum:{}'.format(np.sum([i for j, i in carol_cluster_count][-10:])))
print('Per image: {}'.format(np.sum([i / 102. for j, i in carol_cluster_count][-10:])))
print('Percentage: {}'.format(np.sum([i / 22028. * 100 for j, i in carol_cluster_count][-10:])))

print('**********************Darshan****************************')
print([i for j, i in darshan_cluster_count][-10:])
print([i / 60. for j, i in darshan_cluster_count][-10:])
print([i / 9261. * 100 for j, i in darshan_cluster_count][-10:])
print('Sum:{}'.format(np.sum([i for j, i in darshan_cluster_count][-10:])))
print('Per image: {}'.format(np.sum([i / 60. for j, i in darshan_cluster_count][-10:])))
print('Percentage: {}'.format(np.sum([i / 9261. * 100 for j, i in darshan_cluster_count][-10:])))

print('**********************Diana****************************')
print([i for j, i in diana_cluster_count][-10:])
print([i / 102. for j, i in diana_cluster_count][-10:])
print([i / 39966. * 100 for j, i in diana_cluster_count][-10:])
print('Sum:{}'.format(np.sum([i for j, i in diana_cluster_count][-10:])))
print('Per image: {}'.format(np.sum([i / 102. for j, i in diana_cluster_count][-10:])))
print('Percentage: {}'.format(np.sum([i / 39966. * 100 for j, i in diana_cluster_count][-10:])))

print()
