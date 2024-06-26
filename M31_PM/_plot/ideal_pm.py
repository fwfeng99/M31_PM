import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.style.use('seaborn-whitegrid')


def match_box_id(points_x, points_y, bins_x, bins_y):
    # bins_x_id and bins_y_id
    bins_x_id = np.digitize(points_x, bins_x) - 1
    bins_y_id = np.digitize(points_y, bins_y) - 1

    return bins_x_id, bins_y_id


def cal_point_model(res_xy_list, bins_x, bins_y, bins_num, point_sum_bins, point_sum_values_x, point_sum_values_y):

    for points_inf in res_xy_list:
        points_x = points_inf[0, :]
        points_y = points_inf[1, :]
        point_value_x = points_inf[2, :]
        point_value_y = points_inf[3, :]

        bins_x_id, bins_y_id = match_box_id(points_x, points_y, bins_x, bins_y)

        point_bins, _, _ = np.histogram2d(bins_x_id, bins_y_id, bins=(range(bins_num + 1), range(bins_num + 1)))
        point_values_x, _, _ = np.histogram2d(bins_x_id, bins_y_id, bins=(range(bins_num + 1), range(bins_num + 1)),
                                              weights=point_value_x)
        point_values_y, _, _ = np.histogram2d(bins_x_id, bins_y_id, bins=(range(bins_num + 1), range(bins_num + 1)),
                                              weights=point_value_y)

        point_sum_bins += point_bins
        point_sum_values_x += point_values_x
        point_sum_values_y += point_values_y

    return point_sum_bins, point_sum_values_x, point_sum_values_y


# ------------------------------------------------------------------------------------------------------------------------------------------
# read data
field_idnex = 1
path = 'M31_PM/PM_calculate/Data/PM_result_ideal/'
filename = f'res_ideal_{field_idnex}.csv'
data = pd.read_csv(os.path.join(path, filename))

ra_ideal = data['ra_ideal']
dec_ideal = data['dec_ideal']
pm_ra_ideal = data['ra_pm_ideal'] * np.cos(dec_ideal / 180 * np.pi)
pm_dec_ideal = data['dec_pm_ideal']
# plt.plot(ra_ideal, dec_ideal, '.', markersize=1)

# ------------------------------------------------------------------------------------------------------------------------------------------
# plot histogram figure for PM

fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(1, 2, 1)
plt.hist(pm_ra_ideal, bins=100,
         color='blue', alpha=0.5, edgecolor='black', linewidth=1,
         label='ideal ra PM')

xlabel_text = r'$\mathbf{\mu}_\mathbf{\alpha*}$ [$\mathbf{\mu}$as yr$^{-1}$]'
ax.set_xlabel(xlabel_text, weight='bold', fontsize=14)
ax.set_ylabel('Numbers', fontweight='bold', fontsize=14)
ax.tick_params(axis='x', labelsize=12, width=2)
ax.tick_params(axis='y', labelsize=12, width=2)
ax.text(0.1, 0.9, '(' + 'a' + ')', transform=ax.transAxes, fontsize=12, fontweight='bold')
# plt.legend(loc='upper right')

ax = fig.add_subplot(1, 2, 2)
plt.hist(pm_dec_ideal, bins=100,
         color='red', alpha=0.5, edgecolor='black', linewidth=1,
         label='ideal dec PM')

xlabel_text = r'$\mathbf{\mu}_\mathbf{\delta}$ [$\mathbf{\mu}$as yr$^{-1}$]'
ax.set_xlabel(xlabel_text, weight='bold', fontsize=14)
ax.set_ylabel('Numbers', fontweight='bold', fontsize=14)
ax.tick_params(axis='x', labelsize=12, width=2)
ax.tick_params(axis='y', labelsize=12, width=2)
ax.text(0.1, 0.9, '(' + 'b' + ')', transform=ax.transAxes, fontsize=12, fontweight='bold')

ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
# plt.legend(loc='upper right')

path_fig = 'M31_PM/_plot/figure/'
filename_fig = 'PM_ideal.jpg'
plt.savefig(os.path.join(path_fig, filename_fig), dpi=300)

print(np.std(pm_ra_ideal))
print(np.std(pm_dec_ideal))

# ------------------------------------------------------------------------------------------------------------------------------------------
# plot 2D_vector for PM in each block

# create bins
# bins_num = 2
# bins_x = np.linspace(np.min(ra_ideal), np.max(ra_ideal), bins_num + 1)
# bins_y = np.linspace(np.min(dec_ideal), np.max(dec_ideal), bins_num + 1)
#
# # bins' centers of all box
# bin_centers_x = (bins_x[:-1] + bins_x[1:]) / 2
# bin_centers_y = (bins_y[:-1] + bins_y[1:]) / 2
# bin_centers = np.transpose([np.tile(bin_centers_x, len(bin_centers_y)), np.repeat(bin_centers_y, len(bin_centers_x))])
#
# point_sum_bins = np.zeros((bins_num, bins_num))
# point_sum_values_x = np.zeros((bins_num, bins_num))
# point_sum_values_y = np.zeros((bins_num, bins_num))
#
# res_xy_list = [np.array(data).T]
# [point_sum_bins, point_sum_values_x, point_sum_values_y] = cal_point_model(res_xy_list,
#                                                                            bins_x, bins_y, bins_num,
#                                                                            point_sum_bins,
#                                                                            point_sum_values_x, point_sum_values_y)
#
# point_mean_values_x = np.divide(point_sum_values_x, point_sum_bins,
#                                 out=np.zeros_like(point_sum_values_x), where=point_sum_bins != 0)
# point_mean_values_y = np.divide(point_sum_values_y, point_sum_bins,
#                                 out=np.zeros_like(point_sum_values_y), where=point_sum_bins != 0)
#
# point_mean_values_x = point_mean_values_x.T.flatten()
# point_mean_values_y = point_mean_values_y.T.flatten()
#
# fig, ax = plt.subplots()
# ax.scatter(bin_centers[:, 0], bin_centers[:, 1], s=10)
# ax.quiver(bin_centers[:, 0], bin_centers[:, 1],
#           point_mean_values_x, point_mean_values_y,
#           color='red', scale=80, scale_units='width')
#
# ax.set_xlim([12.2, 12.38])
# ax.set_ylim([42.675, 42.815])
#
# ax.set_xlabel('Ra', fontweight='bold')
# ax.set_ylabel('Dec', fontweight='bold')
#
# plt.show()
# path = 'D:/repos/PycharmProjects/M31_kinematics/M31_PM_diff_correct/_plot/figure/'
# plt.savefig(path + 'PM_ideal_2D_2bins.png', dpi=600)

# ------------------------------------------------------------------------------------------------------------------------------------------

