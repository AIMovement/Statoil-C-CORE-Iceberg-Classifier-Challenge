#!/usr/bin/env python3

import sys
import os
import json
import lzma


# If you want to re-run this you need to download the data sets from the kaggle
# site, https://www.kaggle.com/c/statoil-iceberg-classifier-challenge/data ,
# unpack, and place the .json files in this foler.
def main():
    train_set = load_json('train.json')
    test_set  = load_json('test.json')

    # print keys in the data sets
#      print(train_set[0].keys())
#      print(test_set[0].keys())

    os.makedirs(os.path.join('train_data', 'icebergs'))
    os.makedirs(os.path.join('train_data', 'other'))
    for (k, img) in enumerate(train_set):
        print('Training image {} of {}'.format(k, len(train_set)))

        if img['is_iceberg']:
            fname = os.path.join('train_data',
                                 'icebergs',
                                  img['id'] + '.json')
        else:
            fname = os.path.join('train_data',
                                 'other',
                                  img['id'] + '.json')

        with open(fname, 'w') as fp:
            fp.write(json.dumps({'band_1':    img['band_1'],
                                 'band_2':    img['band_2'],
                                 'inc_angle': img['inc_angle']}))

    os.makedirs('test_data')
    for (k, img) in enumerate(test_set):
        print('Test image {} of {}'.format(k, len(test_set)))

        fname = os.path.join('test_data', img['id'] + '.json')

        with open(fname, 'w') as fp:
            fp.write(json.dumps({'band_1':    img['band_1'],
                                 'band_2':    img['band_2'],
                                 'inc_angle': img['inc_angle']}))

    return True


def load_json(filename):
    with open(filename, 'r') as fp:
        return json.load(fp)


if __name__ == "__main__":
        sys.exit(not main())
