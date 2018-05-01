import numpy as np
import matlab.engine as me
import matlab as m
import csv
import os
import xml.etree.ElementTree as eT


# def main():
#     path_to_images = "/Users/akhil/Desktop/Disease_Images/combined/"
#     gaze_path = '/Users/akhil/Desktop/All_Gaze_Data/'
#     xml_path = "/Users/akhil/Downloads/TPS_Annotations/"
#     img_list = os.listdir(path_to_images)
#     xml_files = os.listdir(xml_path)
#     reference_image = 'CXR204_IM-0683-1001'
#     gaze_dict = gaze_extractor(gaze_path, img_list)
#     xml_dict = dict_creator(xml_path, xml_files)
#     warp_dict = warping(xml_dict, gaze_dict, reference_image, gaze_path, save=True)


def gaze_extractor(path, list):
    '''
    Extracts the gaze data from csv files and stores them in a dictionary with image name and gaze data as key value
    pair.
    :param path: Path string to the folder that has the csv files.
    :param list: list of image names in the folder that has csv files.
    :return: Dictionary with image name and gaze data as key value pair.
    '''
    gaze_dict = {}
    for img in list:
        with open(path + img[:-4] + '.csv', 'r') as f:
            read = csv.reader(f, delimiter=',')
            gaze = []
            for row in read:
                gaze.append([int(row[0]), int(row[1])])
            gaze_dict[img[:-4]] = gaze
    return gaze_dict


def coordinate_xml(file):
    '''
    Extracts a list of annotated coordinates from the xml files that has marked annotations from labelme website.
    :param file: Xml file path that has the marked annotations of each image seperately.
    :return: List of points that represent the annotation on a particular image.
    '''
    points_list = []
    tree = eT.parse(file)
    root = tree.getroot()
    for child in root.iter('object'):
        # name = child.find('name').text
        for poly in child.iter('polygon'):
            for pt in poly.iter('pt'):
                x = float(pt.find('x').text)
                y = float(pt.find('y').text)
                points_list.append([x, y])
    return points_list


def dict_creator(path, files_list):
    '''
    Creates a dictionary with image name and annotated coordinates as key value pair.
    :param path: Path string to the folder that has the xml files with annotations.
    :param files_list: list of all the xml file names in the folder.
    :return: dictionary with image name and list of annotatied coordinates as key value pair.
    '''
    diction = {}
    for file_ in files_list:
        img_name = file_[:-4].upper()
        diction[img_name] = coordinate_xml(path + file_)
    return diction


def warping(anno_dic, gaze_dic, ref_img_name, gaze_path=None, save=False):
    '''
    Warps the gaze coordinates of the required image to the plane of a reference image using Thin Plate Spline
    transformation and saves the resultant coordinates in a new csv file.
    :param anno_dic: dictionary with image names and list of annotated coordinates as key value pair.
    :param gaze_dic: dictionary with image names and list of gaze coordinates as key value pair.
    :param ref_img_name: Reference image name as string.
    :param gaze_path: string that gives the path to the gaze files
    :param save: boolean to save the output warped gaze data to a csv file.
    :return: None
    '''
    warped_dict = {}
    eng = me.start_matlab()
    ref_annos = m.double(np.transpose(np.array(anno_dic[ref_img_name])).tolist())
    for img in anno_dic:
        annotations = m.double(np.transpose(np.array(anno_dic[img])).tolist())
        if np.shape(np.array(annotations))[1] == np.shape(np.array(ref_annos))[1]:
            gaze = m.double(np.transpose(np.array(gaze_dic[img])).tolist())
            tps = eng.tpaps(annotations, ref_annos)
            warp = eng.fnval(tps, gaze)
            warp_ = np.transpose(np.array(warp)).tolist()
            warp_ = [[int(point[0]), int(point[1])] for point in warp_]
            if save is True:
                with open(gaze_path + 'output_warped/' + img + '.csv', 'wb') as cs:
                    writer = csv.writer(cs, delimiter=',')
                    for point in warp_:
                        writer.writerow(point)
            warped_dict[img] = warp_
    # print('Done Warping!')
    return warped_dict

