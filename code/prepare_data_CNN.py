import os
from os.path import dirname, abspath
import matplotlib.pyplot as plt
from config import args

projDIR = dirname(dirname(abspath(__file__)))
img_list = [s.upper().split('.')[0] for s in os.listdir(projDIR + '/gaze/')]

for img in img_list:
    radiologist_name = img.split('_')[0]
    if radiologist_name.upper() == 'CAROL':
        img = plt.imread(args.proj_dir + '/images/CXR129_IM-0189-1001.jpg')[:, :, 0]
        img[:, :, 1:] = 0

print()