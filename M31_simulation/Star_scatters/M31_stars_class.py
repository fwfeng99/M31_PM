import random
import numpy as np
import pandas as pd

import os
import sys

import matplotlib.pyplot as plt

sys.path.extend(['D:\\repos\\PycharmProjects\\M31_kinematics_validate', 'D:/repos/PycharmProjects/M31_kinematics_validate'])
import function


def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)


class M31_Sim(object):

    def __init__(self):
        pass

    def pos_sample(self, r_down: float, r_up: float, stars_num: int, lam: float, ang_down: float, ang_up: float):
        # sample the radius
        r = np.random.uniform(r_down, r_up, size=[stars_num, 1])
        q = np.multiply(r, np.exp(-1 / lam * r))  # targeted function

        k = 0.8  # threshold (> max(q))
        u = np.random.uniform(0, 1.0, size=[stars_num, 1]) * k

        radius = r[q > u]
        # print(len(radius))
        # sample the angle
        angle = np.random.uniform(ang_down, ang_up, size=len(radius))

        #
        x = np.multiply(radius, np.cos(angle))
        y = np.multiply(radius, np.sin(angle))

        #
        star = pd.DataFrame({'radius': radius, 'angle': angle, 'x': x, 'y': y})
        return star


# position distribution-------------------------------------------------------------------------------------------------
set_seed(seed=1)
m31_sim = M31_Sim()

min_r = 21
max_r = 31
stars_num = 3260000
lam = 5.5  # kpc
area_index = 8
ang_range = np.arctan(10 / 25)
ang_mid = -0.88 + np.pi / 2 * (area_index - 4)  # 1~4: 0; 5~8: -0.88
field_angle_down = ang_mid - ang_range
field_angle_up = ang_mid + ang_range

star = m31_sim.pos_sample(min_r, max_r, stars_num, lam, field_angle_down, field_angle_up)

path = "M31_simulation/Star_scatters/Data"
filename = f"M31_disk_{area_index}.csv"
star.to_csv(path_or_buf=os.path.join(path, filename), index=False)

# Parameter prediction-------------------------------------------------------------------------------------------------
# set_seed(seed=1)
# m31_sim = M31_Sim()
# min_r = 10
# max_r = 60
# stars_num = 1000000
# lam = 5.5  # kpc
# field_angle_down = -np.pi
# field_angle_up = np.pi
#
# star = m31_sim.pos_sample(min_r, max_r, stars_num, lam, field_angle_down, field_angle_up)
#
# test_field_coord = [11.250, 42.480]
# [12.28574129, 42.75043505], [11.22870306, 40.95362697], [9.15566471, 39.76631034], [10.09915259, 41.6017492]
# [12.064, 41.865], [10.331, 40.195], [9.185, 40.435], [11.250, 42.480]
# [xi_field, eta_field, ra_field, dec_field, index] = function.local_cel_ideal_coord(star['x'], star['y'],
#                                                                                    785, 73.7, 38.3,
#                                                                                    10.6847083, 41.26875,
#                                                                                    test_field_coord, 0.128)
# print(len(xi_field))
# plt.plot(xi_field, eta_field, '.', color='blue', alpha=0.5)
#
# ang_dist = np.arctan2(star['y'][index], star['x'][index])
# print(np.mean(ang_dist), np.min(ang_dist), np.max(ang_dist))
