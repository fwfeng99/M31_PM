import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy.io import fits


def delect_row(data, dirty_id):
    """
    删除数据中不需要的行
    :param data: 输入的数据
    :param dirty_id: 要去除的行id
    :return: 返回新的塑性后的数据
    """
    new_data = []  # 新建一个列表
    for temp_id in range(len(data)):
        if temp_id not in dirty_id:
            new_data.append(data[temp_id])  # 将不删的数据赋给另一个列表
    return new_data


# get mci fits files' paths-----------------------------------------------------------------------------------------------------------------

in_path = r'D:\repos\PycharmProjects\M31_kinematics\M31_star_magnitude\Data\HST'
# path = unicode(path,'utf-8')  # if file name include chinese
in_filename_list = os.listdir(in_path)

# read *.log -------------------------------------------------------------------------------------------------------------------------------

in_filename = in_filename_list[1]

in_path_filename = in_path + '\\' + in_filename
with open(in_path_filename, mode='r', encoding='utf-8') as f:
    data = f.readlines()
    f.close()

# data clean
dirty_id_1 = list(range(18))
data = delect_row(data, dirty_id_1)

#
data_list = []

for i in data:
    temp = re.split(r'\s+', i)
    temp = temp[len(temp) - 9:-1]
    data_list.append(temp)

column = ['x_pixel', 'y_pixel', 'ra', 'dec', 'm606', 'e606', 'm814', 'e814']
data_clean = pd.DataFrame(data_list, columns=column)

# ------------------------------------------------------------------------------------------------------------------------------------------
in_path = 'D:\\repos\PycharmProjects\M31_kinematics\M31_star_magnitude\Data\HST'
filename = 'hst_disk_magnitude.csv'

hst_disk = pd.read_csv(in_path + '\\' + filename)

ra = hst_disk['ra']
dec = hst_disk['dec']
m606 = hst_disk['m606']
ra_deg = np.zeros(len(ra))
dec_deg = np.zeros(len(dec))

for index, i in enumerate(ra):
    temp = re.split(r':', i)
    temp = list(map(float, temp))
    temp_ra = 15 * temp[0] + 15 * temp[1] / 60 + 15 * temp[2] / 3600
    ra_deg[index] = temp_ra

for index, i in enumerate(dec):
    temp = re.split(r':', i)
    temp = list(map(float, temp))
    temp_deg = temp[0] + temp[1] / 60 + temp[2] / 3600
    dec_deg[index] = temp_deg

disk = SkyCoord('00 49 08.6 +42 45 02', unit=(u.hourangle, u.deg),
                obstime=Time('J2000'))
mci_disk_ra_dec = [disk.ra.degree, disk.dec.degree]

#
fig = plt.figure(num=1, figsize=(6, 5))
ax = fig.add_subplot(1, 1, 1)
plt.plot(ra_deg, dec_deg, '.', markersize=1, c='b')
plt.plot(disk.ra.degree, disk.dec.degree, '.', markersize=1, c='r')
# circle = plt.Circle((disk.ra.degree, disk.dec.degree), 0.04, color='red', linewidth=2)
# ax.add_artist(circle)

len(dec_deg)

fig = plt.figure(num=1, figsize=(8, 5))
ax = fig.add_subplot(1, 1, 1)

ax.set_xlabel('mag', fontweight='bold')
ax.set_ylabel('objects\' number', fontweight='bold')
ax.set_title('Deep Optical Photometry of Disk by Filter F606W', fontweight='bold')

plt.hist(m606, bins=200)
