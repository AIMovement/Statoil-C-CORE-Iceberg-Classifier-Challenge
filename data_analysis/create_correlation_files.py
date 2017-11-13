#!/usr/bin/env python3

import sys
import os
import glob
import json
import numpy as np
import pandas as pd

""" Creating .csv-files necessary to perform the correlation analysis of the TRAIN data 
    in the correlation_analysis.ipynb notebook. Requires data to have been washed. 
    Included: sum of bands, min of bands and max of bands and the correlation to 
    inclination angle. """

def main():
    iceberg_dir = '../data/train_data/icebergs'
    no_iceberg_dir = '../data/train_data/other'

    iceberg_files = get_file_paths(iceberg_dir)
    no_iceberg_files = get_file_paths(no_iceberg_dir)

    ice_img_prop = []
    for (i, iceberg) in enumerate(iceberg_files):
        print('Iceberg image {} of {}'.format(i, len(iceberg_files)))
        fp = load_json(iceberg)
        ice_img_prop.append(get_img_prop(fp))

    iceberg_df = pd.DataFrame(ice_img_prop, columns=['inc_angle',
                                                    'sum_band1', 'min_band1', 'max_band1',
                                                    'sum_band2', 'min_band2', 'max_band2'])

    os.makedirs('../data/image_properties')
    iceberg_df.to_csv(path_or_buf='../data/image_properties/iceberg.csv', sep=',')

    no_ice_img_prop = []
    for (j, no_iceberg) in enumerate(no_iceberg_files):
        print('No iceberg image {} of {}'.format(j, len(no_iceberg_files)))
        fp = load_json(no_iceberg)
        no_ice_img_prop.append(get_img_prop(fp))

    no_iceberg_df = pd.DataFrame(no_ice_img_prop, columns=['inc_angle',
                                                            'sum_band1', 'min_band1', 'max_band1',
                                                            'sum_band2', 'min_band2', 'max_band2'])

    no_iceberg_df.to_csv(path_or_buf='../data/image_properties/no_iceberg.csv', sep=',')

    return True

def load_json(filename):
    with open(filename, 'r') as fp:
        return json.load(fp)

def get_file_paths(dir):
    files = os.path.join(dir,'*.json')
    return glob.glob(files)

def get_img_prop(fp):
    return [fp['inc_angle'],
            np.sum(fp['band_1']), np.min(fp['band_1']), np.max(fp['band_1']),
            np.sum(fp['band_2']), np.min(fp['band_2']), np.max(fp['band_2'])]

if __name__ == "__main__":
        sys.exit(not main())
