import random
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import pickle
from astropy import units as u
from astropy import constants as const
from astropy.coordinates import SkyCoord

import datetime
import sys

sys.path.extend(['D:\\repos\\PycharmProjects\\M31_kinematics_validate', 'D:/repos/PycharmProjects/M31_kinematics_validate'])
import function


def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)


def add_mag_error_chip(xi_eta_mci_field, xi_eta_gmag_error, focal_length, pixel_size):
    temp = xi_eta_mci_field * (180 / np.pi) * 3600 * 1000  # unit is mas
    chip = np.random.normal(loc=temp, scale=xi_eta_gmag_error, size=len(temp))
    chip = chip / (180 / np.pi) / 3600 / 1000
    pixel_obs = chip * focal_length / pixel_size
    return pixel_obs


def add_cte_error_chip(read_coord_size, move_size, star_num, star_mag, star_coord, cte_cof_up, cte_cof_down):
    read_coord_mid = int(read_coord_size / 2)

    star_coord_read = np.ones(star_num)
    star_coord_read[:] = star_coord[:] + move_size

    index = star_coord_read > read_coord_mid
    star_coord_read[index] = read_coord_size - star_coord_read[index]

    star_mag_f = np.array([star_mag ** 0, star_mag ** 1, star_mag ** 2, star_mag ** 3, star_mag ** 4])
    star_mag_f_up = star_mag_f[:, index]
    star_mag_f_down = star_mag_f[:, ~index]

    cte_err_up = (cte_cof_up @ star_mag_f_up) * star_coord_read[index]
    cte_err_down = (cte_cof_down @ star_mag_f_down) * star_coord_read[~index]

    star_coord_err = np.ones(star_num)
    star_coord_err[index] = star_coord[index] + np.squeeze(cte_err_up)
    star_coord_err[~index] = star_coord[~index] - np.squeeze(cte_err_down)

    return star_coord_err


def add_distortion_error_chip(x_pixel, y_pixel, cof):
    r = (x_pixel ** 2 + y_pixel ** 2) ** 0.5
    r_group = np.array([r ** 2, r ** 4, r ** 6])

    x_pixel_obs = x_pixel * (1 + cof @ r_group)
    y_pixel_obs = y_pixel * (1 + cof @ r_group)

    return x_pixel_obs, y_pixel_obs


def trans_to_ref_chip(x_obs_mci_chip_cat, y_obs_mci_chip_cat, trans_model_list, num_years, num_stars):
    x_obs_trans = np.zeros(num_years * num_stars)
    y_obs_trans = np.zeros(num_years * num_stars)

    for i in range(num_years):
        fitting_coefficient = trans_model_list[i]

        x_temp = x_obs_mci_chip_cat[i * num_stars: (i + 1) * num_stars]
        y_temp = y_obs_mci_chip_cat[i * num_stars: (i + 1) * num_stars]

        x_y_trans = np.dot(fitting_coefficient, np.array([x_temp, y_temp, np.ones(len(x_temp))]))

        x_obs_trans[i * num_stars: (i + 1) * num_stars] = x_y_trans[0, :]
        y_obs_trans[i * num_stars: (i + 1) * num_stars] = x_y_trans[1, :]

    return x_obs_trans, y_obs_trans


def chip_to_ra_dec_obs_1(x_obs_mci_chip_cat, y_obs_mci_chip_cat, chip_model_list, years, num_years, num_stars, focal_length, pixel_size):
    ra_obs = np.zeros(len(years) * num_stars)
    dec_obs = np.zeros(len(years) * num_stars)

    for i in range(num_years):
        fitting_coefficient = chip_model_list[i]

        x_temp = x_obs_mci_chip_cat[i * num_stars: (i + 1) * num_stars]
        y_temp = y_obs_mci_chip_cat[i * num_stars: (i + 1) * num_stars]

        xi_eta_obs_cel = np.dot(fitting_coefficient, np.array([x_temp, y_temp, np.ones(len(x_temp))])) / (
                focal_length / pixel_size)

        [ra_obs_temp, dec_obs_temp] = function.ideal_cel_coord(xi_eta_obs_cel[0, :], xi_eta_obs_cel[1, :],
                                                                             mci_field_ra_dec[0], mci_field_ra_dec[1])
        ra_obs[i * num_stars: (i + 1) * num_stars] = ra_obs_temp
        dec_obs[i * num_stars: (i + 1) * num_stars] = dec_obs_temp

    return ra_obs, dec_obs


def chip_to_ra_dec_obs_2(x_pixel_obs, y_pixel_obs, chip_model, years, num_years, num_stars, focal_length, pixel_size):
    ra_obs = np.zeros(len(years) * num_stars)
    dec_obs = np.zeros(len(years) * num_stars)

    for i in range(num_years):
        fitting_coefficient = chip_model

        x_temp = np.array(x_pixel_obs[i * num_stars: (i + 1) * num_stars])
        y_temp = np.array(y_pixel_obs[i * num_stars: (i + 1) * num_stars])

        xi_eta_obs_cel = np.dot(fitting_coefficient, np.array([x_temp, y_temp, np.ones(len(x_temp))])) / (
                focal_length / pixel_size)

        [ra_obs_temp, dec_obs_temp] = function.ideal_cel_coord(xi_eta_obs_cel[0, :], xi_eta_obs_cel[1, :],
                                                                             mci_field_ra_dec[0], mci_field_ra_dec[1])
        ra_obs[i * num_stars: (i + 1) * num_stars] = ra_obs_temp
        dec_obs[i * num_stars: (i + 1) * num_stars] = dec_obs_temp

    return ra_obs, dec_obs


def cal_ideal_PM(ra, dec, years, num_years, num_stars):
    year_lag = years[-1] - years[0]

    ra_t0_cat = ra[0:num_stars]
    dec_t0_cat = dec[0:num_stars]

    ra_t_end_cat = ra[year_lag * num_stars: (year_lag + 1) * num_stars]
    dec_t_end_cat = dec[year_lag * num_stars: (year_lag + 1) * num_stars]

    u_ra_ideal = (ra_t_end_cat - ra_t0_cat) / year_lag
    u_dec_ideal = (dec_t_end_cat - dec_t0_cat) / year_lag

    u_ra_ideal = u_ra_ideal * 3600 * 1000000  # unit is uas
    u_dec_ideal = u_dec_ideal * 3600 * 1000000

    return u_ra_ideal, u_dec_ideal


def cal_obs_PM(ra_obs, dec_obs, years, num_years, num_stars, mid_num):
    years = years - years[mid_num]

    ra_t0_obs = np.zeros(num_stars)
    dec_t0_obs = np.zeros(num_stars)
    u_ra_obs = np.zeros(num_stars)
    u_dec_obs = np.zeros(num_stars)

    for i in range(num_stars):
        ra_t = np.array([ra_obs[i + j * num_stars] for j in range(num_years)])
        dec_t = np.array([dec_obs[i + j * num_stars] for j in range(num_years)])

        # fitting_coefficient = observed_vector / coefficient_matrix

        coefficient_matrix = np.array([[len(years), np.sum(years)], [np.sum(years), np.sum(years * years)]])

        vector_obs_ra = np.array([[np.sum(ra_t)], [np.sum(years * ra_t)]])
        vector_obs_dec = np.array([[np.sum(dec_t)], [np.sum(years * dec_t)]])

        fitting_coefficient_ra = np.dot(np.linalg.inv(coefficient_matrix), vector_obs_ra)
        fitting_coefficient_dec = np.dot(np.linalg.inv(coefficient_matrix), vector_obs_dec)

        ra_t0_obs[i] = fitting_coefficient_ra[0][0]
        dec_t0_obs[i] = fitting_coefficient_dec[0][0]
        u_ra_obs[i] = fitting_coefficient_ra[1][0]
        u_dec_obs[i] = fitting_coefficient_dec[1][0]

    u_ra_obs = u_ra_obs * 3600 * 1000000  # unit is uas
    u_dec_obs = u_dec_obs * 3600 * 1000000

    return ra_t0_obs, dec_t0_obs, u_ra_obs, u_dec_obs


set_seed(2)

# local data--------------------------------------------------------------------------------------------------------------------------------

path = 'M31_simulation/Star_scatters/Data/'
filename = 'M31_star_scatters_mci_disk.csv'
star = pd.read_csv(path + filename)

x_local = star['x']
y_local = star['y']
radius_local = star['radius']
angle_local = star['angle']

# M31 information---------------------------------------------------------------------------------------------------------------------------
M31_c = SkyCoord.from_name('M31')
# M31_c = SkyCoord('00 42 44.330 +41 16 07.50', unit=(u.hourangle, u.deg))
ra_m31_c = M31_c.ra.degree
dec_m31_c = M31_c.dec.degree

m31_distance = 785  # kpc
m31_inclination = 73.7
m31_position_angle = 38.3

# disk information
disk = SkyCoord('00 49 08.6 +42 45 02', unit=(u.hourangle, u.deg))
mci_field_ra_dec = [disk.ra.degree, disk.dec.degree]

# mic instrumental information
MCI_field_size = 0.128  # unit is degree
focal_length = 41253
pixel_size = 0.01
pixel_resolution = 50  # unit is mas

# 00 42 44.330 	+41 16 07.50

# stars in MCI view field-------------------------------------------------------------------------------------------------------------------
[xi_mci_field, eta_mci_field, ra_mci_field, dec_mci_field, mci_field_index_flag] = function.local_cel_ideal_coord(
    x_local, y_local,
    m31_distance, m31_inclination, m31_position_angle,
    ra_m31_c, dec_m31_c,
    mci_field_ra_dec, MCI_field_size)

# angular velocity--------------------------------------------------------------------------------------------------------------------------
path = 'M31_simulation/Rotation_curve/Data/'
filename = 'M31_rotation_curve.csv'
rotation_curve = pd.read_csv(path + filename)

rotation_curve_r = rotation_curve['radius']
rotation_angle_v = (rotation_curve['Vrot'] * 1000) / (rotation_curve['radius'] * const.kpc.value)  # rad/s  help(const)/const.kpc
rotation_curve['rotation_angle_v'] = rotation_angle_v

mci_field_radius_local = radius_local[mci_field_index_flag]
mci_field_angle_local = angle_local[mci_field_index_flag]
mci_field_angle_v = function.disk_rot_ang_v(mci_field_radius_local, rotation_curve_r, rotation_angle_v)

# generate ideal catalogue------------------------------------------------------------------------------------------------------------------
years = np.array([i for i in range(10, 19)])
num_years = len(years)
num_stars = len(xi_mci_field)
columns = ['xi', 'eta', 'ra', 'dec']

mci_field_cat = np.zeros([len(years) * num_stars, len(columns)])

for year, i in zip(years, range(num_years)):
    mci_field_angle_local_year = np.array(mci_field_angle_local) + mci_field_angle_v * 86400 * 365.25 * year
    mci_field_x_local_year = mci_field_radius_local * np.cos(mci_field_angle_local_year)
    mci_field_y_local_year = mci_field_radius_local * np.sin(mci_field_angle_local_year)

    [xi_mci_field, eta_mci_field, ra_mci_field, dec_mci_field,
     mci_field_index_flag] = function.local_cel_ideal_coord(
        mci_field_x_local_year, mci_field_y_local_year,
        m31_distance, m31_inclination, m31_position_angle,
        ra_m31_c, dec_m31_c,
        mci_field_ra_dec, MCI_field_size)

    mci_field_cat[i * num_stars: (i + 1) * num_stars, 0] = xi_mci_field  # xi
    mci_field_cat[i * num_stars: (i + 1) * num_stars, 1] = eta_mci_field  # eta
    mci_field_cat[i * num_stars: (i + 1) * num_stars, 2] = ra_mci_field  # ra
    mci_field_cat[i * num_stars: (i + 1) * num_stars, 3] = dec_mci_field  # dec

# index = pd.MultiIndex.from_product([years, np.arange(len(xi_mci_field))])
mci_field_cat = pd.DataFrame(mci_field_cat, columns=columns)

# add error to generate observed catalogue--------------------------------------------------------------------------------------------------
star_time = datetime.datetime.now()
res_plan_1 = []
res_plan_2 = []

xi_mci_field_cat = np.array(mci_field_cat['xi'])
eta_mci_field_cat = np.array(mci_field_cat['eta'])
ra_mci_field_cat = np.array(mci_field_cat['ra'])
dec_mci_field_cat = np.array(mci_field_cat['dec'])

path = 'M31_stars/HST/Data/'
filename = 'hst_disk_magnitude.csv'
hst_magnitude_cat = pd.read_csv(path + '/' + filename)
mag_mci_field = np.random.choice(hst_magnitude_cat['m606'], num_stars) - 0.169

# 2, add random magnitude error into chip by 10 percent of MCI pixel resolution
pixel_resolution_factor = 0.1
xi_error_neg = np.ones(num_stars) * (pixel_resolution * pixel_resolution_factor)
eta_error_neg = np.ones(num_stars) * (pixel_resolution * pixel_resolution_factor)

# 3, add systematic magnitude error into chip (CTE)
# choose the readout direction is y
read_coord_size = 9216
cte_cof_up = np.array([5e-6, 5e-7, 5e-9, 5e-10, 5e-12]).reshape(1, -1) * 0
cte_cof_down = np.array([5e-6, 5e-7, 5e-9, 5e-10, 5e-12]).reshape(1, -1) * 0  # 0.2 pixel
move_size = int(read_coord_size / 2)
eta_err_list = []

# 4, add systematic distortion error into chip
distortion_cof = np.array([5e-13, 2e-19, 2e-25]) * 0  # 10 pixel
distort_err_list_x = []
distort_err_list_y = []

for model_list_num in range(500):
    in_path = 'M31_PM/Chip_model/Data/validate_model_50_test/'

    with open(in_path + 'chip_model_list_pixel/' + 'chip_model_list_pixel_' + str(model_list_num) + '.pkl', 'rb') as f:
        chip_model_list_pixel = pickle.load(f, encoding='iso-8859-1')

    with open(in_path + 'transform_model_list/' + 'trans_model_list_' + str(model_list_num) + '.pkl', 'rb') as f:
        trans_model_list = pickle.load(f, encoding='iso-8859-1')

    # for i in range(len(chip_model_list_pixel)):
    #     chip_model_list_pixel[i] = np.array([[1, 0, 0], [0, 1, 0]])

    columns = ['pixel_x_obs', 'pixel_y_obs']
    mci_chip_cat = np.zeros([len(years) * num_stars, len(columns)])

    for i in range(len(years)):
        xi_temp = xi_mci_field_cat[i * num_stars: (i + 1) * num_stars]
        eta_temp = eta_mci_field_cat[i * num_stars: (i + 1) * num_stars]

        # 1, add random magnitude error into chip by 10 percent of MCI pixel resolution
        x_year_obs = add_mag_error_chip(xi_temp, xi_error_neg, focal_length, pixel_size)
        y_year_obs = add_mag_error_chip(eta_temp, eta_error_neg, focal_length, pixel_size)

        # 2, add systematic magnitude error into chip (CTE), choose the readout direction is y
        y_year_obs_cte = add_cte_error_chip(read_coord_size, move_size, num_stars, mag_mci_field, y_year_obs,
                                            cte_cof_up, cte_cof_down)
        # eta_err_list.append(np.max(y_year_obs - y_year_obs_cte))

        # 3, add systematic distortion error into chip
        [x_year_obs_distort, y_year_obs_cte_distort] = add_distortion_error_chip(x_year_obs, y_year_obs_cte, distortion_cof)
        # distort_err_list_x.append(np.max(x_year_obs - x_year_obs_distort))
        # distort_err_list_y.append(np.max(y_year_obs_cte - y_year_obs_cte_distort))

        mci_chip_cat[i * num_stars: (i + 1) * num_stars, 0] = x_year_obs_distort  #
        mci_chip_cat[i * num_stars: (i + 1) * num_stars, 1] = y_year_obs_cte_distort  #

    mci_chip_cat = pd.DataFrame(mci_chip_cat, columns=columns)

    # ideal PM------------------------------------------------------------------------------------------------------------------------------
    [u_ra_ideal, u_dec_ideal] = cal_ideal_PM(ra_mci_field_cat, dec_mci_field_cat,
                                             years, num_years, num_stars)

    # plan 1 PM-----------------------------------------------------------------------------------------------------------------------------
    x_obs_mci_chip_cat = np.array(mci_chip_cat['pixel_x_obs'])
    y_obs_mci_chip_cat = np.array(mci_chip_cat['pixel_y_obs'])

    [ra_obs_1, dec_obs_1] = chip_to_ra_dec_obs_1(x_obs_mci_chip_cat, y_obs_mci_chip_cat,
                                                 chip_model_list_pixel,
                                                 years, num_years, num_stars,
                                                 focal_length, pixel_size)

    [ra_t0_obs_1, dec_t0_obs_1, u_ra_obs_1, u_dec_obs_1] = cal_obs_PM(ra_obs_1, dec_obs_1,
                                                                      years, num_years,
                                                                      num_stars, mid_num=0)

    # plan 2 PM-----------------------------------------------------------------------------------------------------------------------------

    # [x_obs_trans, y_obs_trans] = trans_to_ref_chip(mci_chip_cat, trans_model_list, years, num_years, num_stars)

    [x_obs_trans, y_obs_trans] = trans_to_ref_chip(x_obs_mci_chip_cat, y_obs_mci_chip_cat,
                                                   trans_model_list,
                                                   num_years, num_stars)

    mid_num = int((len(chip_model_list_pixel) - 1) / 2)
    chip_model = chip_model_list_pixel[mid_num]
    [ra_obs_2, dec_obs_2] = chip_to_ra_dec_obs_2(x_obs_trans, y_obs_trans,
                                                 chip_model,
                                                 years, num_years, num_stars,
                                                 focal_length, pixel_size)

    [ra_t0_obs_2, dec_t0_obs_2, u_ra_obs_2, u_dec_obs_2] = cal_obs_PM(ra_obs_2, dec_obs_2,
                                                                      years, num_years,
                                                                      num_stars, mid_num=mid_num)

    # validate----------------------------------------------------------------------------------------------------------------------------------
    # print('the first plan\'s result:')
    # print('ideal ra PM:', np.mean(u_ra_ideal), 'observed ra PM:', np.mean(u_ra_obs_1))
    # print('ra PM error:', np.mean(u_ra_ideal) - np.mean(u_ra_obs_1))
    #
    # print('ideal dec PM:', np.mean(u_dec_ideal), 'observed dec PM:', np.mean(u_dec_obs_1))
    # print('dec PM error:', np.mean(u_dec_ideal) - np.mean(u_dec_obs_1))
    #
    # print('the second plan\'s result:')
    # print('ideal ra PM:', np.mean(u_ra_ideal), 'observed ra PM:', np.mean(u_ra_obs_2))
    # print('ra PM error:', np.mean(u_ra_ideal) - np.mean(u_ra_obs_2))
    #
    # print('ideal dec PM:', np.mean(u_dec_ideal), 'observed dec PM:', np.mean(u_dec_obs_2))
    # print('dec PM error:', np.mean(u_dec_ideal) - np.mean(u_dec_obs_2))
    #
    # print('ideal ra PM uncertainty:', np.std(u_ra_ideal) / np.sqrt(num_stars), 'observed ra PM uncertainty:',
    #       np.std(u_ra_obs_2) / np.sqrt(num_stars))
    # print('ideal dec PM uncertainty:', np.std(u_dec_ideal) / np.sqrt(num_stars), 'observed dec PM uncertianty:',
    #       np.std(u_dec_obs_2) / np.sqrt(num_stars))

    res_plan_1.append([np.mean(u_ra_ideal), np.mean(u_ra_obs_1), np.mean(u_dec_ideal), np.mean(u_dec_obs_1),
                       np.std(u_ra_ideal) / np.sqrt(num_stars), np.std(u_ra_obs_1) / np.sqrt(num_stars),
                       np.std(u_dec_ideal) / np.sqrt(num_stars), np.std(u_dec_obs_1) / np.sqrt(num_stars)])

    res_plan_2.append([np.mean(u_ra_ideal), np.mean(u_ra_obs_2), np.mean(u_dec_ideal), np.mean(u_dec_obs_2),
                       np.std(u_ra_ideal) / np.sqrt(num_stars), np.std(u_ra_obs_2) / np.sqrt(num_stars),
                       np.std(u_dec_ideal) / np.sqrt(num_stars), np.std(u_dec_obs_2) / np.sqrt(num_stars)])

columns = ['ra_pm_ideal', 'ra_pm_obs', 'dec_pm_ideal', 'dec_pm_obs', 'ra_pm_ideal_uncer', 'ra_pm_obs_uncer', 'dec_pm_ideal_uncer',
           'dec_pm_obs_uncer']

res_plan_1 = pd.DataFrame(res_plan_1, columns=columns)
res_plan_2 = pd.DataFrame(res_plan_2, columns=columns)

in_path = 'M31_PM/PM_calculate/Data/PM_result/validate_model_50_test/'
res_plan_1.to_csv(in_path + '/' + 'res_plan_1.csv', index=False)
res_plan_2.to_csv(in_path + '/' + 'res_plan_2.csv', index=False)

# res_plan = pd.DataFrame({'ra_pm_ideal': u_ra_ideal, 'dec_pm_ideal': u_dec_ideal})
# res_plan.to_csv('M31_PM/PM_calculate/Data/PM_result/' + 'res_ideal.csv', index=False)

end_time = datetime.datetime.now()
print(end_time - star_time)

# ------------------------------------------------------------------------------------------------------------------------------------------

data = res_plan_1

u_ra_ideal = data['ra_pm_ideal']
u_ra_ideal_err = data['ra_pm_ideal_uncer']
u_ra_obs = data['ra_pm_obs']
u_ra_obs_err = data['ra_pm_obs_uncer']
u_dec_ideal = data['dec_pm_ideal']
u_dec_ideal_err = data['dec_pm_ideal_uncer']
u_dec_obs = data['dec_pm_obs']
u_dec_obs_err = data['dec_pm_obs_uncer']

print(np.mean(u_ra_ideal) - np.mean(u_ra_obs))
print(np.mean(u_dec_ideal) - np.mean(u_dec_obs))
print(np.std(u_ra_obs))
print(np.std(u_dec_obs))

#
# ------------------------------------------------------------------------------------------------------------------------------------------
# histogram
# fig = plt.figure(figsize=(12, 5))
# ax = fig.add_subplot(1, 2, 1)
# ax.set_xlabel(r'ideal ra PM $\mu$as / year', fontweight='bold')
# ax.set_ylabel('numbers', fontweight='bold')
# plt.hist(u_ra_ideal, bins=250)
#
# ax = fig.add_subplot(1, 2, 2)
# ax.set_xlabel('ra PM (add magnitude error) $\mu$as / year', fontweight='bold')
# ax.set_ylabel('numbers', fontweight='bold')
# plt.hist(u_ra_obs, bins=250, color='red')
#
# fig = plt.figure(figsize=(12, 5))
# ax = fig.add_subplot(1, 2, 1)
# ax.set_xlabel(r'ideal dec PM $\mu$as / year', fontweight='bold')
# ax.set_ylabel('numbers', fontweight='bold')
# plt.hist(u_dec_ideal, bins=250)
#
# ax = fig.add_subplot(1, 2, 2)
# ax.set_xlabel('dec PM (add magnitude error) $\mu$as / year', fontweight='bold')
# ax.set_ylabel('numbers', fontweight='bold')
# plt.hist(u_dec_obs, bins=250, color='red')
#
# plt.show()
#
# fig = plt.figure(figsize=(6, 5))
# ax = fig.add_subplot(1, 1, 1)
# plt.hist(m606, bins=150)
# ax.set_xlabel('star magnitude', fontweight='bold')
# ax.set_ylabel('numbers', fontweight='bold')


plt.hist(u_ra_ideal, bins=100, color='yellow', alpha=0.5, edgecolor='green', linewidth=2)
