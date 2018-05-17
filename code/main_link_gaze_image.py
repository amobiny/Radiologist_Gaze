import cv2
import h5py
import matplotlib.pyplot as plt
from utils import *
from gazeWarp import *
import operator
import cPickle as pickle
from config import args
from classifier import *


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
#
#
#
# # match the gaze with extracted features and save the corresponding feature vectors for each image
# feat_dict = {}
# for image_name, value in res_img_gaze_dict.items():
#     all_feats = np.zeros((0, 416))
#     gaze = value[1]
#     feat = value[-1]
#     for x, y in gaze:
#         if x < 256 and y < 256:
#             all_feats = np.concatenate((all_feats, feat[x, y, :].reshape(1, 416)), axis=0)
#     feat_dict[image_name] = all_feats
#
# with open('image_extracted_features.pkl', 'wb') as output:
#     pickle.dump(feat_dict, output, -1)

with open('image_extracted_features.pkl', 'rb') as inputs:
    img_dict = pickle.load(inputs)
print('Data loaded')
#
# feat_img_name = []
# data = np.zeros((0, 416))
# for image_name, feats in img_dict.items():
#     data = np.concatenate((data, feats), axis=0)
#     feat_img_name.extend([image_name for i in range(feats.shape[0])])
#
# # # save data (#samples, #features) and the corresponding name of samples (of length #samples)
# h5f = h5py.File('DATA.h5', 'w')
# h5f.create_dataset('data', data=data)
# h5f.create_dataset('feat_img_name', data=feat_img_name)
# h5f.close()

h5f = h5py.File('DATA.h5', 'r')
data = h5f['data'][:]
feat_img_name = list(h5f['feat_img_name'][:])
h5f.close()

n_clusters = 100
kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=0, max_iter=1000, n_init=10).fit(data)
print('Done with Clustering')

pred_dict = {}
for image_name, feats in img_dict.items():
    labels = kmeans.fit_predict(feats)
    pred_dict[image_name] = labels

print('Creating histograms...')
img_hist_dict = {}
for image_name, labels in pred_dict.items():
    unq_lbl, lbl_count = np.unique(labels, return_counts=True)
    img_hist_dict[image_name] = np.zeros(n_clusters)
    for i in range(len(unq_lbl)):
        img_hist_dict[image_name][unq_lbl[i]] = lbl_count[i]
    img_hist_dict[image_name] /= np.sum(lbl_count)

with open('img_hist_data_100).pkl', 'wb') as output:
    pickle.dump(img_hist_dict, output, -1)
print('Histograms created and saved')







print()
