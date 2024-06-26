import datetime
import random
import numpy as np
import pandas as pd

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

import pickle

import os
import sys

sys.path.extend(['D:\\Code\\Python\\M31_Kinematics', 'D:/Code/Python/M31_Kinematics'])
import function


# ------------------------------------------------------------------------------------------------------------------------
def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)


def readonly_property(attr_name):
    def getter(self):
        return getattr(self, attr_name)

    def setter(self, value):
        raise AttributeError(f"Cannot change {attr_name}")

    return property(getter, setter)


class MCI(object):
    def __init__(self):
        self._mci_field_size = 0.128  # unit is degree
        self._focal_length = 41253  # unit is mm
        self._pix_size = 0.01  # unit is mm
        self._pix_resolution = 50  # unit is mas
        self._f_p = self._focal_length / self._pix_size

    mci_field_size = readonly_property('_mci_field_size')
    focal_length = readonly_property('_focal_length')
    pix_size = readonly_property('_pix_size')
    pix_resolution = readonly_property('_pix_resolution')
    f_p = readonly_property('_f_p')


class MCI_Field(MCI):
    def __init__(self, field_ra_list, field_dec_list, field_index):
        super().__init__()
        self.ra_list = field_ra_list
        self.dec_list = field_dec_list
        self.index = field_index
        self.coord = np.array([field_ra_list[field_index - 1], field_dec_list[field_index - 1]])
        self.M31_cent = SkyCoord.from_name('M31')

    def get_field_ra_dec_list(self):
        mci_field_size_part = self.mci_field_size / 2
        mci_field_size_part_rad = mci_field_size_part * (np.pi / 180)

        coord_list = [function.ideal_cel_coord(x * mci_field_size_part_rad / 2, y * mci_field_size_part_rad / 2,
                                               self.coord[0], self.coord[1]) for x, y in
                      [(-1, -1), (1, -1), (-1, 1), (1, 1)]]

        return coord_list


class MCI_OBS(object):
    def __init__(self, years, num_years, year_0, obs_dt, obs_times):
        self.years = years
        self.num_years = num_years
        self.year_0 = year_0
        self.obs_dt = obs_dt
        self.obs_times = obs_times

    def calculate_observation_times(self):
        years = self.years
        num_years = self.num_years
        year_0 = self.year_0
        obs_dt = self.obs_dt
        obs_times = self.obs_times

        observation_times = []
        for i in range(num_years):
            year = years[i]
            start_time = year_0 + i * obs_dt
            end_time = start_time + obs_times * 3600
            observation_times.append((start_time, end_time))

        return observation_times

    def generate_observation_data(self, data, observation_times):
        observation_data = []
        for start_time, end_time in observation_times:
            observation_data.append(data[(data['time'] >= start_time) & (data['time'] <= end_time)])

        return observation_data


class Cat_process(object):
    def __init__(self, cat_file, cat_col):
        self.cat_file = cat_file
        self.cat_col = cat_col
        self.num_stars = len(self.cat_file)

    def cal_cat_err(self):
        cat_err_f = 1 / 6.6
        num_stars = len(self.cat_file)
        self.ra_err = np.random.normal(loc=np.zeros(num_stars), scale=self.cat_file['e_ra'], size=num_stars) * cat_err_f
        self.dec_err = np.random.normal(loc=np.zeros(num_stars), scale=self.cat_file['e_dec'], size=num_stars) * cat_err_f
        self.pm_ra_err = np.random.normal(loc=np.zeros(num_stars), scale=self.cat_file['e_pm_ra'], size=num_stars) \
                         / np.cos(self.cat_file['dec'] * np.pi / 180) * cat_err_f
        self.pm_dec_err = np.random.normal(loc=np.zeros(num_stars), scale=self.cat_file['e_pm_dec'], size=num_stars) * cat_err_f

    def cal_mag_err(self, MCI_instance):
        pix_res_f = 0.05
        pix_res = MCI_instance.pix_resolution
        self.xi_sigma = np.ones(self.num_stars) * (pix_res * pix_res_f)
        self.eta_sigma = np.ones(self.num_stars) * (pix_res * pix_res_f)  # unit is mas

    def cal_real_pm(self):
        self.pm_ra_real = self.cat_file['pm_ra'] / np.cos(self.cat_file['dec'] * np.pi / 180) + self.pm_ra_err
        self.pm_dec_real = self.cat_file['pm_dec'] + self.pm_dec_err


class Model(object):
    def __init__(self, mci_field, mci_obs, cat_pro):
        self.field = mci_field
        self.epoch = mci_obs
        self.cat = cat_pro

    def cal_cel(self, ra_dec, pm, year):
        return ra_dec + pm * year / 3600 / 1000  # unit is degree

    def add_cat_err_cel(self, ra_dec, ra_dec_err, pm_err, year):
        # return ra_dec + (ra_dec_err + pm_err * year) / 3600 / 1000  # unit is degree
        return (ra_dec * 3600 * 1000 + ra_dec_err + year * pm_err) / 3600 / 1000  # unit is degree

    def add_mag_err_chip(self, xi_eta_cel, xi_eta_sigma):
        temp = xi_eta_cel * (180 / np.pi) * 3600 * 1000  # unit is mas
        xi_eta_chip = np.random.normal(loc=temp, scale=xi_eta_sigma, size=len(temp))
        return xi_eta_chip / (180 / np.pi) / 3600 / 1000 * self.field.f_p

    def cal_chip_one_model(self, xi_obs, eta_obs, x_obs, y_obs, ):
        # xi = a * x + b * y + c
        obs_data_y = xi_obs * self.field.f_p
        obs_data_x = np.array([x_obs, y_obs, np.ones_like(x_obs)])
        obs_num = len(obs_data_x)
        fit_cof_xi = function.polynomial_least_square(obs_data_x, obs_data_y, obs_num)

        # eta = d * x + e * y + f
        obs_data_y = eta_obs * self.field.f_p
        obs_data_x = np.array([x_obs, y_obs, np.ones_like(x_obs)])
        fit_cof_eta = function.polynomial_least_square(obs_data_x, obs_data_y, obs_num)

        fit_cof = np.concatenate((fit_cof_xi, fit_cof_eta), axis=1).T
        return fit_cof

    def cal_chip_cat_model(self):
        chip_model_list = []
        mci_cel_cat = np.zeros([self.epoch.num_years * self.cat.num_stars, 4])
        mci_chip_cat = np.zeros([self.epoch.num_years * self.cat.num_stars, 2])

        for i, year in enumerate(self.epoch.years):

            # set the observed times in every year
            x_list, y_list = [], []
            num_stars = self.cat.num_stars

            for obs_t in range(obs_times):
                year = year + obs_t * obs_dt

                ra_cat_year = self.cal_cel(self.cat.cat_file['ra'],
                                           self.cat.cat_file['pm_ra'] / np.cos(self.cat.cat_file['dec'] * np.pi / 180), year)
                dec_cat_year = self.cal_cel(self.cat.cat_file['dec'], self.cat.cat_file['pm_dec'], year)

                xi_cat_year, eta_cat_year = function.cel_ideal_coord(ra_cat_year, dec_cat_year,
                                                                     self.field.coord[0], self.field.coord[1])

                # xi corresponds to x and ra, eta corresponds to y and dec
                # 1, add Gaia catalogue error (include magnitude error)
                ra_cat_year_obs = self.add_cat_err_cel(ra_cat_year, self.cat.ra_err, self.cat.pm_ra_err, year)
                dec_cat_year_obs = self.add_cat_err_cel(dec_cat_year, self.cat.dec_err, self.cat.pm_dec_err, year)
                [xi_cel_cat_year_obs, eta_cel_cat_year_obs] = function.cel_ideal_coord(ra_cat_year_obs, dec_cat_year_obs,
                                                                                       self.field.coord[0], self.field.coord[1])

                # 2, add random magnitude error into chip (xi_eta_gmag_err is 0.01_per——constant for lack of figure date)
                x_year = self.add_mag_err_chip(xi_cel_cat_year_obs, self.cat.xi_sigma)
                y_year = self.add_mag_err_chip(eta_cel_cat_year_obs, self.cat.eta_sigma)

                x_list.append(x_year)
                y_list.append(y_year)

            # calculate chip model
            x_year_obs = np.mean(x_list, axis=0)
            y_year_obs = np.mean(y_list, axis=0)

            chip_model = self.cal_chip_one_model(xi_cat_year, eta_cat_year, x_year_obs, y_year_obs)
            chip_model_list.append(chip_model)

            start, end = i * num_stars, (i + 1) * num_stars
            mci_cel_cat[start:end, 0:4] = np.vstack((ra_cat_year, dec_cat_year, xi_cat_year, eta_cat_year)).T
            mci_chip_cat[start:end, 0:2] = np.vstack((x_year_obs, y_year_obs)).T

        cat_properties = {
            'mci_cel_cat': mci_cel_cat,
            'mci_chip_cat': mci_chip_cat,
            'chip_model_list': chip_model_list
        }

        for prop_name, prop_value in cat_properties.items():
            setattr(self, prop_name, prop_value)

    def cal_chip_cat_model_new(self):
        self.epoch.years = self.epoch.years[:-1]
        self.epoch.num_years -= 1
        years = self.epoch.years
        num_stars = self.cat.num_stars
        x_obs = self.mci_chip_cat[:, 0]
        y_obs = self.mci_chip_cat[:, 1]

        chip_model_list = []
        columns = ['ra_cat', 'dec_cat', 'xi_cat', 'eta_cat']
        mci_cel_cat = np.zeros([len(years) * num_stars, len(columns)])

        for i, year in enumerate(years):
            ra_cat_year = self.cal_cel(self.cat.cat_file['ra'], self.cat.pm_ra_obs, year)
            dec_cat_year = self.cal_cel(self.cat.cat_file['dec'], self.cat.pm_dec_obs, year)

            [xi_cat_year, eta_cat_year] = function.cel_ideal_coord(ra_cat_year, dec_cat_year, self.field.coord[0], self.field.coord[1])
            start, end = i * num_stars, (i + 1) * num_stars
            x_year_obs = x_obs[start:end]
            y_year_obs = y_obs[start:end]

            chip_model = self.cal_chip_one_model(xi_cat_year, eta_cat_year, x_year_obs, y_year_obs)
            chip_model_list.append(chip_model)

            mci_cel_cat[start:end, 0:4] = np.vstack((ra_cat_year, dec_cat_year, xi_cat_year, eta_cat_year)).T

        cat_properties = {
            'mci_cel_cat': mci_cel_cat,
            'chip_model_list': chip_model_list
        }

        for prop_name, prop_value in cat_properties.items():
            setattr(self, prop_name, prop_value)

    def chip_cel_obs(self):

        x_obs = self.mci_chip_cat[:, 0]
        y_obs = self.mci_chip_cat[:, 1]
        chip_model_list = self.chip_model_list

        ra_obs = np.zeros_like(x_obs)
        dec_obs = np.zeros_like(y_obs)
        num_stars = self.cat.num_stars
        for i in range(self.epoch.num_years):
            fit_cof = chip_model_list[i]

            x_temp = x_obs[i * num_stars: (i + 1) * num_stars]
            y_temp = y_obs[i * num_stars: (i + 1) * num_stars]

            xi_eta_obs_cel = np.dot(fit_cof, np.array([x_temp, y_temp, np.ones_like(x_temp)])) / self.field.f_p
            [ra_obs_temp, dec_obs_temp] = function.ideal_cel_coord(xi_eta_obs_cel[0, :], xi_eta_obs_cel[1, :],
                                                                   self.field.coord[0], self.field.coord[1])
            ra_obs[i * num_stars: (i + 1) * num_stars] = ra_obs_temp
            dec_obs[i * num_stars: (i + 1) * num_stars] = dec_obs_temp

        return ra_obs, dec_obs

    def up_cat_epoch(self, ra_obs, dec_obs):
        self.cat.ra_obs = np.concatenate((ra_obs, self.cat.cat_file['ra']))
        self.cat.dec_obs = np.concatenate((dec_obs, self.cat.cat_file['dec']))
        self.epoch.years = np.append(self.epoch.years, 0)
        self.epoch.num_years += 1

    def cal_obs_pm(self, mid_num=4):

        years = self.epoch.years - self.epoch.years[mid_num]
        num_stars = self.cat.num_stars
        num_years = self.epoch.num_years

        ra_t0_obs, dec_t0_obs, pm_ra_obs, pm_dec_obs, pm_ra_obs_err, pm_dec_obs_err = np.zeros((6, num_stars))

        for i in range(num_stars):
            y_ra_t = np.array([self.cat.ra_obs[i + j * num_stars] for j in range(num_years)])
            y_dec_t = np.array([self.cat.dec_obs[i + j * num_stars] for j in range(num_years)])

            X_years = np.column_stack((np.ones_like(years), years))

            A_ra = np.linalg.inv(X_years.T @ X_years) @ X_years.T @ y_ra_t
            A_dec = np.linalg.inv(X_years.T @ X_years) @ X_years.T @ y_dec_t

            rss_ra = np.sum((y_ra_t - X_years @ A_ra) ** 2)
            rss_dec = np.sum((y_dec_t - X_years @ A_dec) ** 2)

            df = num_years - X_years.shape[1]

            mse_ra = rss_ra / df
            mse_dec = rss_dec / df

            std_ra = np.sqrt(np.diag(np.linalg.inv(X_years.T @ X_years) * mse_ra))
            std_dec = np.sqrt(np.diag(np.linalg.inv(X_years.T @ X_years) * mse_dec))

            ra_t0_obs[i], pm_ra_obs[i] = A_ra
            dec_t0_obs[i], pm_dec_obs[i] = A_dec
            _, pm_ra_obs_err[i] = std_ra
            _, pm_dec_obs_err[i] = std_dec

        deg_mas = 3600000
        pm_ra_obs = pm_ra_obs * deg_mas  # unit is mas
        pm_dec_obs = pm_dec_obs * deg_mas
        pm_ra_obs_err = pm_ra_obs_err * deg_mas
        pm_dec_obs_err = pm_dec_obs_err * deg_mas

        cat_properties = {
            'ra_t0_obs': ra_t0_obs,
            'dec_t0_obs': dec_t0_obs,
            'pm_ra_obs': pm_ra_obs,
            'pm_dec_obs': pm_dec_obs,
            'pm_ra_obs_err': pm_ra_obs_err,
            'pm_dec_obs_err': pm_dec_obs_err
        }

        for prop_name, prop_value in cat_properties.items():
            setattr(self.cat, prop_name, prop_value)

        # return ra_t0_obs, dec_t0_obs, pm_ra_obs, pm_dec_obs, pm_ra_obs_err, pm_dec_obs_err

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
set_seed(1)
mci = MCI()

# field information
field_index = 1
field_ra_list = np.array([12.28574129, 11.22870306, 9.15566471, 10.09915259])
field_dec_list = np.array([42.75043505, 40.95362697, 39.76631034, 41.6017492])
mci_field = MCI_Field(field_ra_list, field_dec_list, field_index)

# observed information
years = np.array(range(14, 24))
num_years = len(years)
year_0 = 2016.0
obs_dt = 3600 / (365.25 * 86400) * 0
obs_times = 10
mci_obs = MCI_OBS(years, num_years, year_0, obs_dt, obs_times)

# Gaia reference catalog process for real PM and magnitude error
path = 'M31_stars/Gaia/Data/'
filename = f'Gaia_M31_disk.csv'
gaia_m31_cat = pd.read_csv(os.path.join(path, filename))
cat_columns = ['ra', 'dec', 'pm_ra', 'pm_dec', 'e_ra', 'e_dec', 'e_pm_ra', 'e_pm_dec', 'gmag']
cat_pro = Cat_process(gaia_m31_cat, cat_columns)
cat_pro.cal_cat_err()
cat_pro.cal_mag_err(mci)
cat_pro.cal_real_pm()

path = f'M31_PM/Chip_model/Data/field_{field_index}/Model_{obs_times}/'
path_filename_chip = os.path.join(os.getcwd(), path, 'chip_model')
path_filename_trans = os.path.join(os.getcwd(), path, 'trans_model')

os.makedirs(path, exist_ok=True)
os.makedirs(path_filename_chip, exist_ok=True)
os.makedirs(path_filename_trans, exist_ok=True)

model = Model(mci_field, mci_obs, cat_pro)
Chip_list = []
for loop in range(2000):
    # first calculation of chip model
    model.cal_chip_cat_model()
    # apply the first chip model to xy for their celestial coordinate
    ra_obs, dec_obs = model.chip_cel_obs()
    # update Gaia reference catalog
    model.up_cat_epoch(ra_obs, dec_obs)

    model.cal_obs_pm()
    # second calculation of chip model
    model.cal_chip_cat_model_new()
    Chip_list.append(model.chip_model_list)

# ----------------------------------------------------------------------------------------------------------------------


