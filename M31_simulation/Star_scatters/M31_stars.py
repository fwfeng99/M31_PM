import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)


def rejection_sampling(max_r, min_r, stars_num, lam):
    # lam = 5.5
    # max_r = 27.5  # max(mci_disk_radius_local)  # 27.098627799152936
    # min_r = 24.5  # min(mci_disk_radius_local)  # 24.99458111882526

    x = np.random.uniform(min_r, max_r, size=[stars_num, 1])  # 750000
    q = np.multiply(x, np.exp(-1 / lam * x))  # targeted function

    k = 0.8  # threshold (> max(q))
    u = np.random.uniform(0, 1.0, size=[stars_num, 1]) * k

    return x[q > u]


# create data-------------------------------------------------------------------------------------------------------------------------------
set_seed(seed=1)
# field radius
set_seed()
max_r = 31
min_r = 21
stars_num = 3260000
lam = 5.5  # kpc
radius = rejection_sampling(max_r, min_r, stars_num, lam)
print(len(radius))
# field angle
field_angle_up = np.arctan(10 / 25)
field_angle_down = np.arctan(10 / 25)
area_index = 4
angle = np.random.uniform(2 * np.pi - field_angle_down, 2 * np.pi + field_angle_up, size=len(radius)) + (area_index - 1) * np.pi / 2

x = np.multiply(radius, np.cos(angle))
y = np.multiply(radius, np.sin(angle))

# store field data
star = pd.DataFrame({'radius': radius, 'angle': angle, 'x': x, 'y': y})  # 1-d data

path = "M31_simulation/Data/star_scatters/"
path = ''
filename = "M31_star_scatter_mci_disk_" + str(area_index) + ".csv"
star.to_csv(path_or_buf=path + filename, index=False)


# create data-------------------------------------------------------------------------------------------------------------------------------
# set_seed(seed=1)
# max_r = 40
# min_r = 0
# stars_num = 7500000
# lam = 5.5  # kpc
# radius = rejection_sampling(max_r, min_r, stars_num, lam)
# print(len(radius))
# angle = np.random.uniform(0, 2 * np.pi, size=len(radius))
#
# x = np.multiply(radius, np.cos(angle))
# y = np.multiply(radius, np.sin(angle))
#
# star = pd.DataFrame({'radius': radius, 'angle': angle, 'x': x, 'y': y})  # 1-d data
#
# path = "M31_simulation/Data/star_scatters/"
# path = ''
# filename = "M31_star_scatters.csv"
# star.to_csv(path_or_buf=path + filename, index=False)


# ------------------------------------------------------------------------------------------------------------------------------------------

# plot
# fig = plt.figure(figsize=(5, 5))
# ax = fig.add_subplot(1, 1, 1)
# ax.plot(x, y, '.', markersize=0.5, color='b')
# ax.set_xlabel('x / kpc')
# ax.set_ylabel('y / kpc')
# ax.set_title('the stars distribution')
# plt.show()
