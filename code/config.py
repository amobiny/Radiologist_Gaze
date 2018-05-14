import argparse
import numpy as np
from os.path import dirname, abspath

parser = argparse.ArgumentParser()
projDIR = dirname(dirname(abspath(__file__)))

# Required Paths
parser.add_argument('--proj_dir', default=projDIR, help='project folder directory')
parser.add_argument('--reference_image', default='CAROL_CXR129_IM-0189-1001', help='reference image name')
parser.add_argument('--label_csv', default=projDIR + '/labels_nih.csv', help='CSV file containing the labels')
parser.add_argument('--all_gaze_path', default=projDIR + '/gaze/', help='path to all gaze files')
parser.add_argument('--xml_path', default=projDIR + '/annotations/', help='path to xml files')
parser.add_argument('--path_to_images', default=projDIR + '/images', help='path to all images')
parser.add_argument('--path_to_videos', default=projDIR + '/animations/', help='path to saved videos')

#
parser.add_argument('--load_warped_gaze', default=True, help='loads the saved warped gazes')
parser.add_argument('--prepare_data', default=False, help='converts the warped data to sub-sequences if True'
                                                          'otherwise loads the already-saved sub-sequences')
parser.add_argument('--load_img_sub_count', default=True, help='loads the dictionary containing images and '
                                                               'sub-sequence counts (in each cluster)')

#
parser.add_argument('--filter_distance', default=15, help='pixel distance for filtering')
parser.add_argument('--subseq_len', default=27, help='sub-sequence length')
parser.add_argument('--n_cluster', default=350, help='number of clusters in Kmeans')

# Random Forrest Classifier
parser.add_argument('--n_estimators', default=1000, help='The number of trees in the forest')
parser.add_argument('--max_depth', default=5, help='The maximum depth of the tree')
parser.add_argument('--max_features', default=0.11, help='#features to consider when looking for the best split')
parser.add_argument('--num_run', default=1, help='number of times running the classifier to get the avg. accuracy')

# others
parser.add_argument('--numvid', default=350, help='#animation videos to be saved')

args = parser.parse_args()
