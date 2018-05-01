import csv
import os
import numpy as np


def file_name_changer(dir_path, rad_name):
    file_names = os.listdir(dir_path)
    for file_ in file_names:
        os.rename(dir_path + file_, dir_path + rad_name + '_' + file_)
    print('done changing ' + rad_name)


def label_file_editor(file_path, file_name, min_row, max_row, rad_name):
    with open(file_path + file_name, 'r') as f:
        read = csv.reader(f, delimiter=',')
        lines = list(read)
        for i in range(len(lines)):
            if min_row <= i <= max_row:
                img_name = lines[i][0]
                lines[i][0] = rad_name + '_' + img_name
        # mod_lines = list(read)
    with open(file_path + file_name, 'wb') as wb:
        writer = csv.writer(wb)
        writer.writerows(lines)
    print('done')


if __name__ == '__main__':
    # file_name_changer('C:/Users/akhil/Desktop/annotations_combined/diana/', 'diana')
    # file_name_changer('C:/Users/akhil/Desktop/annotations_combined/darshan/', 'darshan')
    # file_name_changer('C:/Users/akhil/Desktop/annotations_combined/carol/', 'carol')
    # with open('C:/Users/akhil/Desktop/' + 'Preprocess_combined2.csv', 'r') as f:
    #     read = csv.reader(f)
    #     lines = list(read)
    # print()
    with open('C:/Users/akhil/Desktop/' + 'labels2.csv') as infile:
        reader = csv.reader(infile)
        reader.next()
        lines = [lin for lin in reader]
    # with open('C:/Users/akhil/Desktop/' + 'Preprocess_combined4.csv', 'wb') as outfile:
    #     writer = csv.writer(outfile, delimiter=',')
    #     writer.writerows(lines)
    #
    #     for line in infile:
    #         splitted = line.split()
    #         outfile.write(" ".join(line.split('\t')).replace(' ', ','))
    #         # outfile.write(",")
    #         outfile.write("\n")
        print()
    # label_file_editor('C:/Users/akhil/Desktop/', 'Preprocess_combined.csv', 1, 102, 'carol')
    # label_file_editor('C:/Users/akhil/Desktop/', 'Preprocess_combined.csv', 103, 204, 'darshan')
    # label_file_editor('C:/Users/akhil/Desktop/', 'Preprocess_combined.csv', 205, 306, 'diana')
    # carol_gaze_path = 'C:/Users/akhil/Desktop/carol_gaze/'
    # file_name_changer(carol_gaze_path, 'carol')
    # diana_gaze_path = 'C:/Users/akhil/Desktop/diana_gaze/'
    # file_name_changer(diana_gaze_path, 'diana')
