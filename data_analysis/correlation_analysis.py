#!/usr/bin/env python
# coding: utf-8

# Python script version of correlation_analysis.ipynb
# If the pyhon script becomes out of sync with the notebook,
# 'jupyter nbconvert --to script correlation_analysis.ipynb'
# can be used to convert it again. Be prepared to do some manual fixes in that
# case.



# # Correlation and distribution analysis


import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import os

PROJ_ROOT = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..')


# Directory and files created after running
# data_analysis/create_correlation_files.py. The files consists of inc. angle
# in addition to the summation, minimum and maximum of pixel values for iceberg
# respectively no iceberg images


dir = os.path.join(PROJ_ROOT, 'data', 'image_properties')
iceberg_df = pd.read_csv(os.path.join(dir, 'iceberg.csv'))
no_iceberg_df = pd.read_csv(os.path.join(dir, 'no_iceberg.csv'))


# Remove N/A entries!
no_iceberg_df = no_iceberg_df.drop(no_iceberg_df[(no_iceberg_df.inc_angle == 'na')].index)
no_iceberg_df.head(3)


# ## First, lets take a look at some distributions for iceberg images vs. no iceberg images...!
# ### INCLINATION ANGLE
fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True, figsize=(12,5))

sns.distplot(iceberg_df['inc_angle'], hist=False, color="g", kde_kws={"shade": True}, ax=ax1)
sns.distplot(no_iceberg_df['inc_angle'], hist=False, color="b", kde_kws={"shade": True}, ax=ax2)

ax1.set_title('Iceberg')
ax2.set_title('No Iceberg')

plt.show()


# ### MIN. PIXEL VALUES
f, axes = plt.subplots(2, 2, figsize=(10, 7))
sns.despine(left=True)

sns.distplot(iceberg_df['min_band1'], kde=True, color="g", kde_kws={"shade": True}, ax=axes[0, 0])
sns.distplot(iceberg_df['min_band2'], kde=True, color="b", kde_kws={"shade": True}, ax=axes[1, 0])
sns.distplot(no_iceberg_df['min_band1'], kde=True, color="r", kde_kws={"shade": True}, ax=axes[0, 1])
sns.distplot(no_iceberg_df['min_band2'], kde=True, color="m", kde_kws={"shade": True}, ax=axes[1, 1])

axes[0,0].set_title('Iceberg')
axes[0,1].set_title('No iceberg')

plt.setp(axes, yticks=[])
plt.tight_layout()
plt.show()


# ### MAX. PIXEL VALUES
f, axes = plt.subplots(2, 2, figsize=(10, 7))
sns.despine(left=True)

sns.distplot(iceberg_df['max_band1'], kde=True, color="g", kde_kws={"shade": True}, ax=axes[0, 0])
sns.distplot(iceberg_df['max_band2'], kde=True, color="b", kde_kws={"shade": True}, ax=axes[1, 0])
sns.distplot(no_iceberg_df['max_band1'], kde=True, color="r", kde_kws={"shade": True}, ax=axes[0, 1])
sns.distplot(no_iceberg_df['max_band2'], kde=True, color="m", kde_kws={"shade": True}, ax=axes[1, 1])

axes[0,0].set_title('Iceberg')
axes[0,1].set_title('No iceberg')

plt.setp(axes, yticks=[])
plt.tight_layout()
plt.show()


# ### Aaand the sums...
f, axes = plt.subplots(2, 2, figsize=(12, 7))
sns.despine(left=True)

sns.distplot(iceberg_df['sum_band1'], kde=True, color="g", kde_kws={"shade": True}, ax=axes[0, 0])
sns.distplot(iceberg_df['sum_band2'], kde=True, color="b", kde_kws={"shade": True}, ax=axes[1, 0])
sns.distplot(no_iceberg_df['sum_band1'], kde=True, color="r", kde_kws={"shade": True}, ax=axes[0, 1])
sns.distplot(no_iceberg_df['sum_band2'], kde=True, color="m", kde_kws={"shade": True}, ax=axes[1, 1])

axes[0,0].set_title('Iceberg')
axes[0,1].set_title('No iceberg')

plt.setp(axes, yticks=[])
plt.tight_layout()
plt.show()


# ## Visualize correlation between inclination angle and the sum of bands
stack_band1 = np.stack([iceberg_df['inc_angle'], iceberg_df['sum_band1']], axis=-1)
df_band1 = pd.DataFrame(stack_band1, columns=['inc_angle', 'sum_band1'])

g = sns.JointGrid(x="inc_angle", y="sum_band1", data=df_band1)
g = g.plot_joint(sns.kdeplot, cmap="BuGn_r")
g = g.plot_marginals(sns.kdeplot, shade=True)

plt.show()


stack_band2 = np.stack([iceberg_df['inc_angle'], iceberg_df['sum_band2']], axis=-1)
df_band2 = pd.DataFrame(stack_band2, columns=['inc_angle', 'sum_band2'])

h = sns.JointGrid(x="inc_angle", y="sum_band2", data=df_band2)
h = h.plot_joint(sns.kdeplot, cmap="RdBu_r")
h = h.plot_marginals(sns.kdeplot, shade=True)

plt.show()
