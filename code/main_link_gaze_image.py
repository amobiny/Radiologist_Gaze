import cv2
import h5py
import matplotlib.pyplot as plt

from my_plots import plot_box
from utils import *
from gazeWarp import *
import operator
import cPickle as pickle
from config import args
from classifier import *
from config import args


# gaze_file_names = os.listdir(args.all_gaze_path)
# raw_gaze_dict = gazeExtractor(gaze_file_names, args.all_gaze_path)
# xml_files = os.listdir(args.xml_path)
#
# filtered_gaze_dict = gazeFilter(raw_gaze_dict, distance=args.filter_distance, check_distance=True)
# print('Done Filtering')


def crop_image_gaze(img, gaze, tol=255):
    # img is image data
    # tol  is tolerance
    mask = img < tol
    border_len = np.argmax(mask[0, :]) - 1
    shifted_gaze = [[gx - border_len, gy] for gx, gy in gaze]
    return img[np.ix_(mask.any(1), mask.any(0))], shifted_gaze


def gaze_img_Plotter(data, image):
    x = []
    y = []
    for point in data:
        x.append(point[0])
        y.append(point[1])
    plt.plot(x, y)
    plt.plot(x[-1], y[-1], 'o')
    plt.imshow(image, cmap='gray')
    plt.show()
    print()


def resize_img_gaze(imag, gaze, size):
    height, width = imag.shape
    x_scale, y_scale = width / size, height / size
    r_img = cv2.resize(imag, dsize=(int(size), int(size)), interpolation=cv2.INTER_CUBIC)
    r_gaze = [[int(gx / x_scale), int(gy / y_scale)] for gx, gy in gaze]
    return r_img, r_gaze


# res_img_gaze_dict = {}
# for image_name, gaze in filtered_gaze_dict.items():
#     radiologist_name = image_name.split('_')[0].upper()
#     img_name = image_name.split(radiologist_name)[1][1:]
#     img = plt.imread(args.proj_dir + '/images/' + img_name + '.jpg')[:, :, 0]
#     new_img, new_gaze = crop_image_gaze(img, gaze)
#     # gaze_img_Plotter(gaze, img)
#     # gaze_img_Plotter(new_gaze, new_img)
#     # Resize the cropped image to (256, 256)
#     res_img, res_gaze = resize_img_gaze(new_img, new_gaze, size=256.)
#     res_info = [res_img, res_gaze]
#     # gaze_img_Plotter(res_gaze, res_img)
#     res_img_gaze_dict[image_name] = res_info
#
# with open('resized_img_gaze_dict.pkl', 'wb') as output:
#     pickle.dump(res_gaze_dict, output, -1)

# with open('resized_img_gaze_dict.pkl', 'rb') as inputs:
#     res_img_gaze_dict = pickle.load(inputs)
# print('Resized images and corresponding gazes loaded')


# h5f = h5py.File('/home/cougarnet.uh.edu/amobiny/Desktop/Radiologist_Gaze/densenet/features_from_densenet.h5', 'r')
# all_feats = h5f['all_feats'][:]
# image_name_list = h5f['image_name_list'][:]
# h5f.close()
# print('DenseNet features loaded')
#
# # resize the features from DenseNet
# res_feats = np.zeros((all_feats.shape[0], 256, 256, all_feats.shape[-1]))
# for i in range(all_feats.shape[0]):
#     for j in range(all_feats.shape[-1]):
#         feat = all_feats[i, :, :, j]
#         res_feats[i, :, :, j] = cv2.resize(feat, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
#
# # save resized features
# h5f = h5py.File('resized_features_from_densenet.h5', 'w')
# h5f.create_dataset('res_feats', data=res_feats)
# h5f.create_dataset('image_name_list', data=image_name_list)
# h5f.close()

# h5f = h5py.File('resized_features_from_densenet.h5', 'r')
# res_feats = h5f['res_feats'][:]
# image_name_list = list(h5f['image_name_list'][:])
# h5f.close()
# print('DenseNet resized-features loaded')
#
#
# # create dictionary of image, gaze, and features
# for image_name, value in res_img_gaze_dict.items():
#     idx = image_name_list.index(image_name)
#     feat = res_feats[idx]
#     res_img_gaze_dict[image_name].append(feat)
#
# with open('img_gaze_feat_dict.pkl', 'wb') as output:
#     pickle.dump(res_img_gaze_dict, output, -1)

# with open('img_gaze_feat_dict.pkl', 'rb') as inputs:
#     res_img_gaze_dict = pickle.load(inputs)
# print('dictionary of image name:(image, gaze, features) loaded')



# match the gaze with extracted features and save the corresponding feature vectors for each image
# 418 is the 416 features + (x, y) positions
# feat_dict = {}
# for image_name, value in res_img_gaze_dict.items():
#     all_feats = np.zeros((0, 418))
#     gaze = value[1]
#     feat = value[-1]
#     for x, y in gaze:
#         if x < 256 and y < 256:
#             pos = np.array([x, y])
#             new_feat = np.concatenate((feat[x, y, :].reshape(1, 416), pos.reshape(1, 2)), axis=1)
#             all_feats = np.concatenate((all_feats, new_feat), axis=0)
#     feat_dict[image_name] = all_feats
#
# with open('image_extracted_features.pkl', 'wb') as output:
#     pickle.dump(feat_dict, output, -1)

# #
# with open('image_extracted_features.pkl', 'rb') as inputs:
#     feat_dict = pickle.load(inputs)
# print('Data loaded; containing: image_name: features (of size: [#samples, 418])')
# #
# # feat_img_name = []
# # data = np.zeros((0, 418))
# # for image_name, feats in feat_dict.items():
# #     data = np.concatenate((data, feats), axis=0)
# #     feat_img_name.extend([image_name for i in range(feats.shape[0])])
# #
# # # save data (#samples, #features) and the corresponding name of samples (of length #samples)
# # h5f = h5py.File('DATA.h5', 'w')
# # h5f.create_dataset('data', data=data)
# # h5f.create_dataset('feat_img_name', data=feat_img_name)
# # h5f.close()
#
#
# h5f = h5py.File('DATA.h5', 'r')
# data = h5f['data'][:]
# feat_img_name = list(h5f['feat_img_name'][:])
# h5f.close()
# #
n_clusters = 500
# kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=0, max_iter=1000, n_init=20).fit(data[:, :416])
# print('Done with Clustering')
# centers = kmeans.cluster_centers_
#
# # add the predicted labels to the dictionary of data
# for image_name, feats in feat_dict.items():
#     labels = kmeans.fit_predict(feats[:, :416])
#     feat_dict[image_name] = np.concatenate((feats, labels.reshape(-1, 1)), axis=1)
# print('Predicted labels added to the dictionary of data (feat_data)')
#
# #-
# print('Creating histograms...')
# img_hist_dict = {}
# for image_name, feat_labels in feat_dict.items():
#     labels = feat_labels[:, -1]
#     unq_lbl, lbl_count = np.unique(labels, return_counts=True)
#     img_hist_dict[image_name] = np.zeros(n_clusters)
#     for i in range(len(unq_lbl)):
#         img_hist_dict[image_name][int(unq_lbl[i])] = lbl_count[i]
#     img_hist_dict[image_name] /= np.sum(lbl_count)
# print('Dictionary of histograms created; image_name: histogram (of size #clusters)')
#
#
# # adding labels to the dictionary
# with open(args.label_csv, 'r') as p:
#     reader = csv.reader(p, delimiter='\t')
#     reader.next()
#     lines = [lin for lin in reader]
#     img_name_list = [l[0].upper() for l in lines]
#
# for i in range(len(img_name_list)):
#     image_name = img_name_list[i]
#     if image_name in img_hist_dict.keys():
#         y = np.array([int(lines[i][j]) for j in range(1, 16)])
#         img_hist_dict[image_name] = np.append(img_hist_dict[image_name], y)
# print('Dictionary of histograms created; image_name: histogram (of size #clusters+15=115), ')
# # 15 labels added to the histograms
#
# # add the images and their corresponding gaze sequence to the img_hist_dict
# for image_name, value in res_img_gaze_dict.items():
#     image = value[0]
#     gaze_data = value[1]
#     img_hist_dict[image_name] = img_hist_dict[image_name] + [image] + [gaze_data]
#
# with open('img_hist_data_100.pkl', 'wb') as output:
#     pickle.dump([img_hist_dict, feat_dict, centers], output, -1)
# print('Histograms created and saved')
# img_hist_dict::: image_name: list of length 3 including:
#                                   [image(256, 256) ,
#                                   hist+label of size n_cluster+15=115,
#                                   gaze_data (list of different lengths including the x,y positions)]
# feat_dict::::::: image_name: features of size: [#samples, 419] where 419 = #features(416) + [x,y] + cluster_prediction
# centers: cluster centers of size (n_cluster=100, dim=416)

# --------------------------------Visualize image, gaze, histogram--------------------------------------


def img_gaze_hist_plot(img, img_name, gaze, hist, label):
    fig, axs = plt.subplots(nrows=1, ncols=2)
    fig.set_size_inches(12, 6)
    ax = axs[0]
    x = []
    y = []
    for point in gaze:
        x.append(point[0])
        y.append(point[1])
    ax.plot(x, y)
    ax.plot(x[-1], y[-1], 'o')
    ax.set_title(img_name)
    # ax.set_xlim(0, 256)
    # ax.set_ylim(0, 256)
    ax.imshow(img, cmap='gray')
    ax = axs[1]
    ax.bar(range(len(hist)), hist)
    conds = ['Cardio. ', 'Infilt. ', 'Nodule ', 'Normal ', 'PlurEff. ', 'Pneumtrx.']
    cond_lbl = np.where(label == 1)[0]
    tit = [conds[a] for a in cond_lbl]
    img_title = '-'.join(tit)
    ax.set_title(img_title)
    plt.show()


def prepare_data(img_dict, n_clust):
    img_name_list = []
    data = np.zeros((0, n_clust))
    label = np.zeros((0, 15))
    for img_name, info in img_dict.items():
        img_name_list.append(img_name)
        data = np.concatenate((data, info[0][:n_clust].reshape(1, -1)), axis=0)
        label = np.concatenate((label, info[0][n_clust:].reshape(1, -1)), axis=0)
    return data, label, img_name_list


with open("img_hist_data_500.pkl", 'rb') as inputs:
    img_hist_dict, feat_dict, centers = pickle.load(inputs)
print('Data loaded')

# ---This part is to plot example images, gaze, and corresponding histograms
# image_number_to_plot = 7
# image_name = img_hist_dict.keys()[image_number_to_plot]
# image = img_hist_dict[image_name][1]
# gaze_data = img_hist_dict[image_name][-1]
# hist_data = img_hist_dict[image_name][0][:n_clusters]
# label = img_hist_dict[image_name][0][n_clusters:n_clusters+6]
# img_gaze_hist_plot(image, image_name, gaze_data, hist_data, label)

# 0.Cardio, 1.Infilt, 2.Nodule, 3.Normal, 4.PlurEff, 5.Pneumtrx, 6.Carol, 7.Darshan, 8.Diana
# 9.Cardio, 10.Infilt, 11.Nodule, 12. Normal, 13.PlurEff, 14.Pneumtrx,

# --------------------------------Start Classification--------------------------------------
x, y, image_name_list = prepare_data(img_hist_dict, n_clusters)
classifier = RandomForestClassifier(n_estimators=args.n_estimators,
                                    max_depth=args.max_depth,
                                    oob_score=True,
                                    max_features=args.max_features)
# x_train, y_train, x_test, y_test = train_test_split(x, y[:, 2])
x_train = x
y_train = y[:, 2]
classifier.fit(x_train, y_train)
result = classifier.predict(x_train)
y_prob_test = classifier.predict_proba(x_train)
feat_imp = classifier.feature_importances_
imp_feat, imp_feat_idx = np.sort(feat_imp), np.argsort(feat_imp)
imp_centers = centers[imp_feat_idx] # in ascending order
cumsum_feat = np.cumsum(np.flip(imp_feat, 0))
# classifier.oob_score_
print('Classifier Trained & Feature importance generated')
accuracy = np.sum(np.equal(np.reshape(result, (-1, 1)),
                           np.reshape(y_train, (-1, 1)))) / np.float(result.size)
print('accuracy= {0:.02%}'.format(accuracy))
# Let's see where is these important features placed on images


def plot_important_features(img_dict, feat_dic, img_name, sorted_centers, sorted_feat_idx, num, n_clusters):
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
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

    ax = axs[1]
    ax.imshow(image, cmap='gray')
    for i in range(len(mask)):
        if mask[i]:
            ax.plot(x[i], y[i], 'o', color='hotpink')
    # ax.set_title(img_name)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

    ax = axs[2]
    ax.plot(x, y)
    for i in range(len(mask)):
        if mask[i]:
            ax.plot(x[i], y[i], 'o', color='hotpink')
    # ax.set_title(img_name)
    ax.imshow(image, cmap='gray')
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

    ax = axs[3]
    baar = ax.bar(range(len(hist)), hist)
    if isinstance(imp_features_idx.tolist(), int):
        baar[imp_features_idx].set_color('hotpink')
    else:
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
image_name = 'DIANA_CXR11_IM-0067-1001'
image_name = 'DIANA_CXR3416_IM-1651-0001-0001'
image_name = 'DIANA_CXR728_IM-2287-1001'
image_name = 'CAROL_CXR3416_IM-1651-0001-0001'
image_name = 'CAROL_CXR1028_IM-0022-1001'
image_name = 'DIANA_CXR1136_IM-0092-1001'
# percentile = np.argmax(cumsum_feat > 0.2)
which_feature = 499  # up to number of clusters
plot_important_features(img_hist_dict, feat_dict, image_name, imp_centers, imp_feat_idx, num=which_feature, n_clusters=500)
# plot_box(img_hist_dict, feat_dict, image_name, imp_centers, imp_feat_idx, num=302)


print()
