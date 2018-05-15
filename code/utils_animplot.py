import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from config import args


def gazePlotter(data, name=None, flattened=True, save=False, output_path=None, plot=True, img_name=None):
    x = []
    y = []
    if not img_name:
        img_name = 'CXR129_IM-0189-1001'
    if flattened:
        for i in range(0, len(data) - 1, 2):
            x.append(data[i])
            y.append(data[i + 1])
    else:
        for point in data:
            x.append(point[0])
            y.append(point[1])
    fig = plt.figure()
    plt.xlim(0, 2560)
    plt.ylim(1440, 0)
    line, = plt.plot(x, y)

    def animate(ii):
        line.set_data(x[:ii + 1], y[:ii + 1])
        return line
    img = plt.imread(args.proj_dir + '/images/' + img_name + '.jpg')
    plt.imshow(img)
    ani = animation.FuncAnimation(fig, animate, frames=50, interval=100)
    if save is True:
        if not os.path.exists(output_path + str(name.split('/')[0])):
            os.makedirs(output_path + str(name.split('/')[0]))
        ani.save(output_path + str(name) + str() + '.mp4', writer='ffmpeg', fps=15)
    if plot is True:
        plt.show()


def gaze_plot_save(cluster_sorted, imp_centers, num=5, path='carol_most_important/'):
    for i in range(1, num+1):
        gazePlotter(imp_centers[cluster_sorted[-i][0]], name=path + str(i) + '_' + str(cluster_sorted[-i][1]),
                    flattened=True, save=True, output_path=args.path_to_videos, plot=False)
        print('Creating and Saving Animation.....')


def rawSubPlotter(filteredGaze, imgName, output_path, save=False, plot=True, name=None):
    gaze_ = filteredGaze[imgName]
    gazePlotter(gaze_, flattened=False, save=save, output_path=output_path, plot=plot, name=name)
