#!/usr/bin/env python3

import os
import sys
import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


train_iceberg_path = os.path.join('..', 'data', 'train_data', 'icebergs')
train_other_path   = os.path.join('..', 'data', 'train_data', 'other')

IMG_SIZE = 75

def main():
    plot_all_in_dir(train_iceberg_path)
    plot_all_in_dir(train_other_path)

    return True


def plot_all_in_dir(directory):
    for f in os.listdir(directory):
        with open(os.path.join(directory, f), 'r') as fp:
            # load image
            img = json.load(fp)

            # create image data
            mat_1 = np.matrix(img['band_1']).reshape(IMG_SIZE, IMG_SIZE)
            mat_2 = np.matrix(img['band_2']).reshape(IMG_SIZE, IMG_SIZE)
            mat_s = mat_1 + mat_2

            # Do some plotting
            fig = plt.figure(figsize=(12, 6))
            fig.suptitle('{}: Is Iceberg: {}: Angle: {}'.
                    format(f, img['is_iceberg'], img['inc_angle']))

            ax = plt.subplot(1, 3, 1)
            ax.imshow(mat_1, cmap=cm.Greys)
            ax.set_title('Band 1')

            ax = plt.subplot(1, 3, 2)
            ax.imshow(mat_2, cmap=cm.Greys)
            ax.set_title('Band 2')

            ax = plt.subplot(1, 3, 3)
            ax.imshow(mat_s, cmap=cm.Greys)
            ax.set_title('Sum of bands')

            plt.show()


if __name__ == '__main__':
    sys.exit(not main())
