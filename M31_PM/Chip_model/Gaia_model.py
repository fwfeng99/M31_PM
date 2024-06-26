import datetime
import random
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.table import Table
from astropy import units as u
from astropy import constants as const
from astropy.coordinates import SkyCoord, ICRS, GCRS, Distance
from astropy.time import Time

import pickle

import sys

sys.path.extend(['D:\\Code\\Python\\M31_Kinematics', 'D:/Code/Python/M31_Kinematics'])
import function


def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)


def add_mag_error_ra_dec(ra_dec, ra_dec_err, pm_ra_dec_err, year):
    temp = ra_dec * 3600 * 1000  # unit is mas
    obs = temp + ra_dec_err + year * pm_ra_dec_err
    # obs = np.random.normal(loc=temp, scale=ra_dec_err, size=len(temp))
    obs = obs / 3600 / 1000  # unit is degree
    return obs


def add_mag_error_chip(xi_eta_mci_field, xi_eta_gmag_error, focal_length, pixel_size):
    temp = xi_eta_mci_field * (180 / np.pi) * 3600 * 1000  # unit is mas
    chip = np.random.normal(loc=temp, scale=xi_eta_gmag_error, size=len(temp))
    # for i in range(len(temp)):
    #     chip[i] = stats.norm.rvs(temp[i], xi_eta_gmag_error[i], 1)

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


def cal_chip_model_pixel(xi_mci_field_obs, eta_mci_field_obs, x_pixel_obs, y_pixel_obs, focal_length, pixel_size):
    # xi = a * x + b * y + c
    observed_data_y = xi_mci_field_obs * (focal_length / pixel_size)
    observed_data_x = np.array([x_pixel_obs, y_pixel_obs, np.ones(len(x_pixel_obs))])
    observed_num = len(observed_data_x)
    fitting_coefficient_xi = function.polynomial_least_square(observed_data_x, observed_data_y, observed_num)

    # eta = d * x + e * y + f
    observed_data_y = eta_mci_field_obs * (focal_length / pixel_size)
    observed_data_x = np.array([x_pixel_obs, y_pixel_obs, np.ones(len(x_pixel_obs))])
    fitting_coefficient_eta = function.polynomial_least_square(observed_data_x, observed_data_y, observed_num)

    fitting_coefficients = np.concatenate((fitting_coefficient_xi, fitting_coefficient_eta), axis=1).T
    return fitting_coefficients


def cal_chip_model_rad(xi_mci_field_obs, eta_mci_field_obs, x_pixel_obs, y_pixel_obs, focal_length, pixel_size):
    # xi = a * x + b * y + c
    x_pixel_obs = x_pixel_obs / (focal_length / pixel_size)
    y_pixel_obs = y_pixel_obs / (focal_length / pixel_size)

    observed_data_y = xi_mci_field_obs
    observed_data_x = np.array([x_pixel_obs, y_pixel_obs, np.ones(len(x_pixel_obs))])
    observed_num = len(observed_data_x)
    fitting_coefficient_xi = function.polynomial_least_square(observed_data_x, observed_data_y, observed_num)

    # eta = d * x + e * y + f
    observed_data_y = eta_mci_field_obs
    observed_data_x = np.array([x_pixel_obs, y_pixel_obs, np.ones(len(x_pixel_obs))])
    fitting_coefficient_eta = function.polynomial_least_square(observed_data_x, observed_data_y, observed_num)

    fitting_coefficients = np.concatenate((fitting_coefficient_xi, fitting_coefficient_eta), axis=1).T
    return fitting_coefficients


def ideal_coordinate_to_chip_coordinate(xi, eta, chip_model, focal_length, pixel_size):
    x_y = np.linalg.inv(chip_model[:2, :2]) @ (np.array([xi, eta]) - chip_model[:, 2].reshape(-1, 1))
    x_y = x_y * focal_length / pixel_size

    return x_y


def cal_trans_model_pixel(x_mid, y_mid, x, y):
    # x_mid = a * x + b * y + c
    # y_mid = d * x + e * y + f

    observed_data_x = np.array([x, y, np.ones_like(x)])
    observed_num = len(observed_data_x)

    observed_data_y = x_mid
    fitting_coefficient_x = function.polynomial_least_square(observed_data_x, observed_data_y, observed_num)
    observed_data_y = y_mid
    fitting_coefficient_y = function.polynomial_least_square(observed_data_x, observed_data_y, observed_num)

    fitting_coefficients = np.concatenate((fitting_coefficient_x, fitting_coefficient_y), axis=1).T
    return fitting_coefficients


set_seed(1)

# read star catalogue ----------------------------------------------------------------------------------------------------------------------
path = 'M31_stars/Gaia/Data/'
filename = 'Gaia_M31_disk_class.csv'
gaia_m31_catalogue = pd.read_csv(path + '/' + filename)

ra_mci_field = np.array(gaia_m31_catalogue['ra'])
dec_mci_field = np.array(gaia_m31_catalogue['dec'])
pm_ra_mci_field = np.array(gaia_m31_catalogue['pm_ra'])
pm_dec_mci_field = np.array(gaia_m31_catalogue['pm_dec'])
ra_mci_field_error = np.array(gaia_m31_catalogue['e_ra'])
dec_mci_field_error = np.array(gaia_m31_catalogue['e_dec'])
pm_ra_mci_field_error = np.array(gaia_m31_catalogue['e_pm_ra'])
pm_dec_mci_field_error = np.array(gaia_m31_catalogue['e_pm_dec'])
gmag_mci_field = np.array(gaia_m31_catalogue['gmag'])

# mci view field information

disk = SkyCoord('00 49 08.6 +42 45 02', unit=(u.hourangle, u.deg), obstime=Time('J2000'))
mci_disk_ra_dec = [disk.ra.degree, disk.dec.degree]
mci_field_size = 0.128  # unit is degree

focal_length = 41253  # unit is mm
pixel_size = 0.01  # unit is mm
pixel_resolution = 50  # unit is mas

M31_c = SkyCoord.from_name('M31')

# columns = ['ra', 'dec', 'e_ra', 'e_dec', 'plx', 'e_plx', 'pm_ra', 'pm_dec', 'e_pm_ra', 'e_pm_dec', 'gmag']
# data = pd.DataFrame(
#     {'ra': ra_mci_field, 'dec': dec_mci_field, 'e_ra': ra_mci_field_error, 'e_dec': dec_mci_field_error,
#      'plx': plx_mci_field, 'e_plx': plx_mci_field_error, 'pm_ra': pm_ra_mci_field, 'pm_dec': pm_dec_mci_field,
#      'e_pm_ra': pm_ra_mci_field_error, 'e_pm_dec': pm_dec_mci_field_error, 'gmag': gmag_mci_field})
# data.to_csv('Gaia_M31_disk_class.csv', index=False)

# ------------------------------------------------------------------------------------------------------------------------------------------

star_time = datetime.datetime.now()

#
years = range(10, 19)
year_0 = 2016.0
num_stars = len(ra_mci_field)

# 2, add random magnitude error into chip by 1 percent of MCI pixel resolution
pixel_resolution_factor = 0.01
xi_gmag_error = np.ones(num_stars) * (pixel_resolution * pixel_resolution_factor)
eta_gmag_error = np.ones(num_stars) * (pixel_resolution * pixel_resolution_factor)  # unit is mas

# 3, add systematic magnitude error into chip (CTE)
# choose the readout direction is y
read_coord_size = 9216
cte_cof_up = np.array([5e-6, 5e-7, 5e-9, 5e-10, 5e-12]).reshape(1, -1) * 0
cte_cof_down = np.array([5e-6, 5e-7, 5e-9, 5e-10, 5e-12]).reshape(1, -1) * 0  # 0.2 pixel
move_size = int(read_coord_size / 2)

# 4, add systematic distortion error into chip
distortion_cof = np.array([5e-13, 2e-19, 2e-25]) * 0  # 10 pixel

for loop in range(2000):

    chip_model_list_pixel = []
    trans_model_list = []

    # Gaia catalogue error
    error_factor = 50
    ra_gmag_error = np.random.normal(loc=np.zeros(num_stars), scale=ra_mci_field_error, size=num_stars) / 6.6 * error_factor
    dec_gmag_error = np.random.normal(loc=np.zeros(num_stars), scale=dec_mci_field_error, size=num_stars) / 6.6 * error_factor
    pm_ra_gmag_error = np.random.normal(loc=np.zeros(num_stars), scale=pm_ra_mci_field_error, size=num_stars) \
                       / np.cos(dec_mci_field * np.pi / 180) / 6.6 * error_factor
    pm_dec_gmag_error = np.random.normal(loc=np.zeros(num_stars), scale=pm_dec_mci_field_error, size=num_stars) / 6.6 * error_factor

    # ------------------------------------------------------------------------------------------------------------------------------------------
    #
    columns = ['ra_obs', 'dec_obs', 'xi_obs', 'eta_obs', 'x_obs', 'y_obs']
    mci_cel_chip_cat = np.zeros([len(years) * num_stars, len(columns)])

    for i, year in enumerate(years):
        # BCRS i = 0; year = years[i]
        ra_mci_field_year = ra_mci_field + year * pm_ra_mci_field / np.cos(dec_mci_field * np.pi / 180) / 1000 / 3600
        dec_mci_field_year = dec_mci_field + year * pm_dec_mci_field / 1000 / 3600

        [xi_mci_field_year, eta_mci_field_year] = function.celestail_to_ideal_coordiante(ra_mci_field_year, dec_mci_field_year,
                                                                                         mci_disk_ra_dec[0], mci_disk_ra_dec[1])

        # xi corresponds to x and ra, eta corresponds to y and dec

        # 1, add Gaia catalogue error (include magnitude error)
        # ra_gmag_error = np.sqrt(ra_mci_field_error ** 2 + (year * pm_ra_mci_field_error) ** 2) / 6.6
        # dec_gmag_error = np.sqrt(dec_mci_field_error ** 2 + (year * pm_dec_mci_field_error) ** 2) / 6.6

        ra_mci_field_year_obs = add_mag_error_ra_dec(ra_mci_field_year, ra_gmag_error, pm_ra_gmag_error, year)
        dec_mci_field_year_obs = add_mag_error_ra_dec(dec_mci_field_year, dec_gmag_error, pm_dec_gmag_error, year)
        [xi_cel_mci_field_year_obs, eta_cel_mci_field_year_obs] = function.celestail_to_ideal_coordiante(ra_mci_field_year_obs,
                                                                                                         dec_mci_field_year_obs,
                                                                                                         mci_disk_ra_dec[0],
                                                                                                         mci_disk_ra_dec[1])

        # 2, add random magnitude error into chip (xi_eta_gmag_error is 0.01_per——constant for lack of figure date)
        x_pixel_year_obs = add_mag_error_chip(xi_mci_field_year, xi_gmag_error, focal_length, pixel_size)
        y_pixel_year_obs = add_mag_error_chip(eta_mci_field_year, eta_gmag_error, focal_length, pixel_size)

        # 3, add systematic magnitude error into chip (CTE), choose the readout direction is y
        y_pixel_year_obs = add_cte_error_chip(read_coord_size, move_size, num_stars, gmag_mci_field, y_pixel_year_obs,
                                              cte_cof_up, cte_cof_down)

        # 4, add systematic distortion error into chip
        [x_pixel_year_obs, y_pixel_year_obs] = add_distortion_error_chip(x_pixel_year_obs, y_pixel_year_obs, distortion_cof)

        # calculate chip model
        chip_model_pixel = cal_chip_model_pixel(xi_cel_mci_field_year_obs, eta_cel_mci_field_year_obs,
                                                x_pixel_year_obs, y_pixel_year_obs,
                                                focal_length, pixel_size)

        chip_model_list_pixel.append(chip_model_pixel)

        #
        mci_cel_chip_cat[i * num_stars: (i + 1) * num_stars, 0] = ra_mci_field_year_obs
        mci_cel_chip_cat[i * num_stars: (i + 1) * num_stars, 1] = dec_mci_field_year_obs
        mci_cel_chip_cat[i * num_stars: (i + 1) * num_stars, 2] = xi_cel_mci_field_year_obs
        mci_cel_chip_cat[i * num_stars: (i + 1) * num_stars, 3] = eta_cel_mci_field_year_obs
        mci_cel_chip_cat[i * num_stars: (i + 1) * num_stars, 4] = x_pixel_year_obs
        mci_cel_chip_cat[i * num_stars: (i + 1) * num_stars, 5] = y_pixel_year_obs

    # ------------------------------------------------------------------------------------------------------------------------------------------
    # diff transform model

    mid_num = int((len(years) - 1) / 2)
    mid_year = year_0 + years[mid_num]

    ra_mci_field_years_obs = mci_cel_chip_cat[:, 0]
    dec_mci_field_years_obs = mci_cel_chip_cat[:, 1]
    x_pixel_mid = mci_cel_chip_cat[mid_num * num_stars: (mid_num + 1) * num_stars, 4]
    y_pixel_mid = mci_cel_chip_cat[mid_num * num_stars: (mid_num + 1) * num_stars, 5]

    for i, year in enumerate(years):
        ra_mci_field_year_obs_align = ra_mci_field_years_obs[i * num_stars: (i + 1) * num_stars] + (mid_num - i) * pm_ra_mci_field \
                                      / np.cos(dec_mci_field * np.pi / 180) / 3600 / 1000
        dec_mci_field_year_obs_align = dec_mci_field_years_obs[i * num_stars: (i + 1) * num_stars] + (mid_num - i) * pm_dec_mci_field \
                                       / 3600 / 1000

        [xi_cel_mci_field_year_obs_align, eta_cel_mci_field_year_obs_align] = function.celestail_to_ideal_coordiante(
            ra_mci_field_year_obs_align, dec_mci_field_year_obs_align, mci_disk_ra_dec[0], mci_disk_ra_dec[1])

        x_y_pixel = np.linalg.inv(chip_model_list_pixel[i][:2, :2]) @ (
                np.array([xi_cel_mci_field_year_obs_align, eta_cel_mci_field_year_obs_align])
                * (focal_length / pixel_size) - chip_model_list_pixel[i][:, 2].reshape(-1, 1))

        trans_model = cal_trans_model_pixel(x_pixel_mid, y_pixel_mid, x_y_pixel[0, :], x_y_pixel[1, :])
        trans_model_list.append(trans_model)

    # ---------------------------------------------------------------------------------------------------------------------------------------
    in_path = 'M31_PM/Chip_model/Data/validate_model_50/'

    with open(in_path + 'chip_model_list_pixel/' + 'chip_model_list_pixel_' + str(loop) + '.pkl', 'wb') as f:
        pickle.dump(chip_model_list_pixel, f)

    with open(in_path + 'transform_model_list/' + 'trans_model_list_' + str(loop) + '.pkl', 'wb') as f:
        pickle.dump(trans_model_list, f)

end_time = datetime.datetime.now()
print(end_time - star_time)

# ------------------------------------------------------------------------------------------------------------------------------------------

# add star magnitude to chip by 5 percent of MCI pixel resolution
# add star magnitude to ra_dec_celestial by Gaia catalogue
# set seed(1)
# improve program running speed
