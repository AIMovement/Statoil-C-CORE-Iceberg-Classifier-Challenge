import os
import pandas as pd
import numpy as np

sub_path = "C:/Users/okarnbla/Downloads/other_kaggle_submission/"
all_files = os.listdir(sub_path)

# Read and concatenate submissions
outs = [pd.read_csv(os.path.join(sub_path, f), index_col=0) for f in all_files]
concat_sub = pd.concat(outs, axis=1)
cols = list(map(lambda x: "is_iceberg_" + str(x), range(len(concat_sub.columns))))
concat_sub.columns = cols
concat_sub.reset_index(inplace=True)

# Check correlation
concat_sub.corr()

# get the data fields ready for stacking
concat_sub['is_iceberg_max'] = concat_sub.iloc[:, 1:6].max(axis=1)
concat_sub['is_iceberg_min'] = concat_sub.iloc[:, 1:6].min(axis=1)
concat_sub['is_iceberg_mean'] = concat_sub.iloc[:, 1:6].mean(axis=1)
concat_sub['is_iceberg_median'] = concat_sub.iloc[:, 1:6].median(axis=1)

# set up cutoff threshold for lower and upper bounds, easy to twist
cutoff_lo = 0.7
cutoff_hi = 0.3

# load the model with best base performance
sub_base = pd.read_csv(sub_path + 'stack_minmax_bestbase.csv')

concat_sub['is_iceberg_base'] = sub_base['is_iceberg']

concat_sub.loc[(concat_sub['is_iceberg_max'] < 0.95) & (concat_sub['is_iceberg_max'] > cutoff_lo), 'is_iceberg_max'] = concat_sub['is_iceberg_max'] + 0.05
concat_sub.loc[(concat_sub['is_iceberg_min'] > 0.05) & (concat_sub['is_iceberg_min'] < cutoff_hi), 'is_iceberg_min'] = concat_sub['is_iceberg_min'] + 0.05

concat_sub['is_iceberg'] = np.where(np.all(concat_sub.iloc[:, 1:6] > cutoff_lo, axis=1),
                                    concat_sub['is_iceberg_max'],
                                    np.where(np.all(concat_sub.iloc[:, 1:6] < cutoff_hi, axis=1),
                                             concat_sub['is_iceberg_min'],
                                             0.40 * concat_sub['is_iceberg_base'] + 0.6 * concat_sub['is_iceberg_median']))

#a = np.where(np.all(concat_sub.iloc[:, 1:6] > cutoff_lo, axis=1),
#                                    concat_sub['is_iceberg_max'],
#                                    np.where(np.all(concat_sub.iloc[:, 1:6] < cutoff_hi, axis=1),
#                                             concat_sub['is_iceberg_min'],
#                                             concat_sub['is_iceberg_base']))


concat_sub[['id', 'is_iceberg']].to_csv('subm_stack.csv',
                                        index=False, float_format='%.6f')

#import scripts.viztools as viz
#viz.dist(concat_sub['is_iceberg'], a)