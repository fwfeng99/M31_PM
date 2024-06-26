import datetime
import random
import numpy as np
import pandas as pd

from astropy import units as u
from astropy import constants as const
from astropy.coordinates import SkyCoord
from astropy.time import Time

import os
import sys

sys.path.extend(['D:\\repos\\PycharmProjects\\M31_kinematics_validate', 'D:/repos/PycharmProjects/M31_kinematics_validate'])
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
    def __init__(self, field_ra_list, field_dec_list, field_index, blocks_num):
        super().__init__()
        self.ra_list = field_ra_list
        self.dec_list = field_dec_list
        self.index = field_index
        self.coord = np.array([field_ra_list[field_index - 1], field_dec_list[field_index - 1]])
        self.M31_c = SkyCoord.from_name('M31')
        self.coord_cel = function.cel_ideal_coord(self.coord[0], self.coord[1], self.M31_c.ra.value, self.M31_c.dec.value)
        self.M31_dis = 785  # kpc
        self.M31_inc = 73.7
        self.M31_PA = 38.3
        self.rho = np.arctan((self.coord_cel[0] ** 2 + self.coord_cel[1] ** 2) ** 0.5)
        self.phi = np.pi - np.arctan2(self.coord_cel[1], self.coord_cel[0])
        self.v_sys = -301

        self.blocks = blocks_num
        blocks_index = np.linspace(-self.blocks + 1, self.blocks - 1, self.blocks)
        blocks_index_list = [[j, i] for i in blocks_index for j in blocks_index]

        mci_field_size_part = self.mci_field_size / self.blocks * (np.pi / 180)
        self.coord_part_list = [function.ideal_cel_coord(x * mci_field_size_part / 2, y * mci_field_size_part / 2,
                                                         self.coord[0], self.coord[1]) for x, y in blocks_index_list]


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

    def cal_cat_err(self, cat_err_f=1 / 6.6):
        num_stars = len(self.cat_file)
        self.ra_err = np.random.normal(loc=np.zeros(num_stars), scale=self.cat_file['e_ra'], size=num_stars) * cat_err_f
        self.dec_err = np.random.normal(loc=np.zeros(num_stars), scale=self.cat_file['e_dec'], size=num_stars) * cat_err_f
        self.pm_ra_err = np.random.normal(loc=np.zeros(num_stars), scale=self.cat_file['e_pm_ra'], size=num_stars) \
                         / np.cos(self.cat_file['dec'] * np.pi / 180) * cat_err_f
        self.pm_dec_err = np.random.normal(loc=np.zeros(num_stars), scale=self.cat_file['e_pm_dec'], size=num_stars) * cat_err_f

    def cal_mag_err(self, MCI_instance, pix_res_f=0.05):
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


class M31_PM(object):
    def __init__(self, mci_field, mci_obs):
        self.field = mci_field
        self.epoch = mci_obs

    def get_star_local(self, path):
        self.star_local = pd.read_csv(path)

    def cal_star_coord(self):

        # position (celestial and ideal)
        [_, _,
         self.ra_field, self.dec_field,
         self.field_flag] = function.local_cel_ideal_coord(self.star_local['x'], self.star_local['y'],
                                                           self.field.M31_dis, self.field.M31_inc, self.field.M31_PA,
                                                           self.field.M31_c.ra.degree, self.field.M31_c.dec.degree,
                                                           self.field.coord, self.field.mci_field_size)
        self.num_stars = len(self.ra_field)

        [self.xi_field, self.eta_field] = function.cel_ideal_coord(self.ra_field, self.dec_field,
                                                                   self.field.M31_c.ra.degree, self.field.M31_c.dec.degree)
        self.rho_field = np.arctan((self.xi_field ** 2 + self.eta_field ** 2) ** 0.5)
        self.phi_field = np.pi - np.arctan2(self.eta_field, self.xi_field)

    def cal_star_v(self):

        i = self.field.M31_inc * np.pi / 180
        rho = self.rho_field
        phi = self.phi_field
        p_t = self.phi_field - self.field.M31_PA * np.pi / 180 - np.pi / 2  # phi substracts theta(PA)
        D0 = self.field.M31_dis
        ra_c = self.field.M31_c.ra.rad
        dec_c = self.field.M31_c.dec.rad
        ra = np.deg2rad(self.ra_field)
        dec = np.deg2rad(self.dec_field)
        v_sys = self.field.v_sys

        vt = 160
        theta = -2
        Vr = -230

        v2 = vt * np.cos(rho) * np.cos(phi - theta) - v_sys * np.sin(rho) + \
             np.sin(i) * np.cos(p_t) * (np.cos(i) * np.sin(rho) + np.sin(i) * np.cos(rho) * np.sin(p_t)) / (
                     np.cos(i) ** 2 * np.cos(p_t) ** 2 + np.sin(p_t) ** 2) ** 0.5 * Vr
        v3 = -vt * np.sin(phi - theta) - (np.cos(i) ** 2 * np.cos(p_t) ** 2 + np.sin(p_t) ** 2) / (
                np.cos(i) ** 2 * np.cos(p_t) ** 2 + np.sin(p_t) ** 2) ** 0.5 * Vr

        cos_T = (np.sin(dec) * np.cos(dec_c) * np.cos(ra - ra_c) - np.cos(dec) * np.sin(dec_c)) / np.sin(rho)
        sin_T = np.cos(dec_c) * np.sin(ra - ra_c) / np.sin(rho)
        D_f = (np.cos(i) * np.cos(rho) - np.sin(i) * np.sin(rho) * np.sin(p_t)) / (D0 * np.cos(i))
        rot_f = np.array([[-sin_T, -cos_T], [cos_T, -sin_T]])

        self.v23 = np.array([v2, v3])
        self.vwn = np.zeros_like(self.v23)
        for i in range(len(v2)):
            self.vwn[:, i] = D_f[i] * rot_f[:, :, i] @ self.v23[:, i] / 4.740470463533348 * 1000

    def cal_cel(self, ra_dec, pm, year):
        return ra_dec + pm * year / 3600 / 1000000  # pm is uas

    def cal_ideal_cat(self):
        years = self.epoch.years
        num_stars = self.num_stars
        columns = ['xi', 'eta', 'ra', 'dec']

        field_cat = np.zeros([len(years) * num_stars, len(columns)])

        # ideal catalogue
        for i, year in enumerate(years):
            ra_field = self.cal_cel(self.ra_field, -self.vwn[0, :] / np.cos(self.dec_field * np.pi / 180), year)
            dec_field = self.cal_cel(self.dec_field, self.vwn[1, :], year)

            [xi_field, eta_field] = function.cel_ideal_coord(ra_field, dec_field,
                                                             self.field.coord[0], self.field.coord[1])

            start, end = i * num_stars, (i + 1) * num_stars
            field_cat[start:end, 0:4] = np.vstack((xi_field, eta_field, ra_field, dec_field)).T

        self.field_ideal_cat = pd.DataFrame(field_cat, columns=columns)

    def cal_ideal_pm(self):
        index_year = -1
        years = self.epoch.years
        num_stars = self.num_stars
        ra = np.array(self.field_ideal_cat['ra'])
        dec = np.array(self.field_ideal_cat['dec'])

        year_lag = years[index_year] - years[0]

        ra_t0_cat = ra[0:num_stars]
        dec_t0_cat = dec[0:num_stars]

        ra_t_end_cat = ra[year_lag * num_stars: (year_lag + 1) * num_stars]
        dec_t_end_cat = dec[year_lag * num_stars: (year_lag + 1) * num_stars]

        self.pm_ra_ideal = (ra_t_end_cat - ra_t0_cat) / year_lag * 3600 * 1000000
        self.pm_dec_ideal = (dec_t_end_cat - dec_t0_cat) / year_lag * 3600 * 1000000
        columns = ['ra_ideal', 'dec_ideal', 'ra_pm_ideal', 'dec_pm_ideal']
        self.ideal_cat = pd.DataFrame(np.array([ra_t0_cat, dec_t0_cat, self.pm_ra_ideal, self.pm_dec_ideal]).T, columns=columns)

    def cal_mag_err(self, pix_res_f=0.15):

        pix_res = 50
        self.xi_sigma = np.ones(self.num_stars) * (pix_res * pix_res_f)
        self.eta_sigma = np.ones(self.num_stars) * (pix_res * pix_res_f)  # unit is mas

    def add_mag_err_chip(self, xi_eta_cel, xi_eta_sigma):
        temp = xi_eta_cel * (180 / np.pi) * 3600 * 1000  # unit is mas
        xi_eta_chip = np.random.normal(loc=temp, scale=xi_eta_sigma, size=len(temp))
        return xi_eta_chip / (180 / np.pi) / 3600 / 1000 * self.field.f_p

    def cal_chip_cat(self):
        years = self.epoch.years
        num_stars = self.num_stars
        columns = ['x_obs', 'y_obs']
        mci_chip_cat = np.zeros([len(years) * num_stars, len(columns)])
        xi_field_cat = np.array(self.field_ideal_cat['xi'])
        eta_field_cat = np.array(self.field_ideal_cat['eta'])

        for i in range(num_years):

            start, end = i * num_stars, (i + 1) * num_stars
            xi_temp = xi_field_cat[start: end]
            eta_temp = eta_field_cat[start: end]

            x_year_list = []
            y_year_list = []

            for _ in range(obs_times):
                # 1, add random magnitude error into chip by 10 percent of MCI pixel resolution
                x_year = self.add_mag_err_chip(xi_temp, self.xi_sigma)
                y_year = self.add_mag_err_chip(eta_temp, self.eta_sigma)

                x_year_list.append(x_year)
                y_year_list.append(y_year)

            mci_chip_cat[start: end, 0] = np.mean(x_year_list, axis=0)
            mci_chip_cat[start: end, 1] = np.mean(y_year_list, axis=0)

        self.field_chip_cat = pd.DataFrame(mci_chip_cat, columns=columns)

    def chip_cel_obs(self, chip_model_list):

        x_obs = self.field_chip_cat['x_obs']
        y_obs = self.field_chip_cat['y_obs']

        ra_obs = np.zeros_like(x_obs)
        dec_obs = np.zeros_like(y_obs)
        num_stars = self.num_stars
        for i in range(self.epoch.num_years):
            fit_cof = chip_model_list[i]

            x_temp = x_obs[i * num_stars: (i + 1) * num_stars]
            y_temp = y_obs[i * num_stars: (i + 1) * num_stars]

            xi_eta_obs_cel = np.dot(fit_cof, np.array([x_temp, y_temp, np.ones_like(x_temp)])) / self.field.f_p
            [ra_obs_temp, dec_obs_temp] = function.ideal_cel_coord(xi_eta_obs_cel[0, :], xi_eta_obs_cel[1, :],
                                                                   self.field.coord[0], self.field.coord[1])
            ra_obs[i * num_stars: (i + 1) * num_stars] = ra_obs_temp
            dec_obs[i * num_stars: (i + 1) * num_stars] = dec_obs_temp

        self.ra_obs = ra_obs
        self.dec_obs = dec_obs

    def cal_obs_pm(self, mid_num=4):

        years = self.epoch.years - self.epoch.years[mid_num]
        num_stars = self.num_stars
        num_years = self.epoch.num_years

        ra_t0_obs, dec_t0_obs, pm_ra_obs, pm_dec_obs, pm_ra_obs_err, pm_dec_obs_err = np.zeros((6, num_stars))

        for i in range(num_stars):
            y_ra_t = np.array([self.ra_obs[i + j * num_stars] for j in range(num_years)])
            y_dec_t = np.array([self.dec_obs[i + j * num_stars] for j in range(num_years)])

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

        deg_mas = 3600000000
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
            setattr(self, prop_name, prop_value)

        # return ra_t0_obs, dec_t0_obs, pm_ra_obs, pm_dec_obs, pm_ra_obs_err, pm_dec_obs_err

    def obs_pm_part(self, res):

        for coord_part, res_part in zip(self.field.coord_part_list, res):
            [xi, eta] = function.cel_ideal_coord(self.ra_field, self.dec_field, coord_part[0], coord_part[1])
            field_size_part = self.field.mci_field_size / self.field.blocks * (np.pi / 180)
            index_flag = (np.abs(xi) < field_size_part / 2) & (np.abs(eta) < field_size_part / 2)

            res_part.append([np.mean(self.pm_ra_ideal[index_flag]), np.mean(self.pm_ra_obs[index_flag]),
                             np.mean(self.pm_dec_ideal[index_flag]), np.mean(self.pm_dec_obs[index_flag])])
            # num_stars_part.append(np.sum(index_flag))


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
set_seed(1)
mci = MCI()

field_index = 1
field_ra_list = np.array([12.28574129, 11.22870306, 9.15566471, 10.09915259, 11.25])
field_dec_list = np.array([42.75043505, 40.95362697, 39.76631034, 41.6017492, 42.48])
blocks_num = 2
mci_field = MCI_Field(field_ra_list, field_dec_list, field_index, blocks_num)

years = np.array(range(14, 24))
num_years = len(years)
year_0 = 2016.0
obs_dt = 3600 / (365.25 * 86400) * 0
obs_times = 10
mci_obs = MCI_OBS(years, num_years, year_0, obs_dt, obs_times)

path = 'M31_stars/Gaia/Data/'
filename = f'Gaia_M31_disk_{field_index}.csv'
gaia_m31_cat = pd.read_csv(os.path.join(path, filename))
cat_columns = ['ra', 'dec', 'pm_ra', 'pm_dec', 'e_ra', 'e_dec', 'e_pm_ra', 'e_pm_dec', 'gmag']
cat_pro = Cat_process(gaia_m31_cat, cat_columns)

cat_err_f = 1 / 6.6
pix_err_f_gaia = 0.05
cat_pro.cal_cat_err(cat_err_f)
cat_pro.cal_mag_err(mci, pix_err_f_gaia)
cat_pro.cal_real_pm()

path = f'M31_PM/Chip_model/Data/field_{field_index}/Model_{obs_times}/'
path_filename_chip = os.path.join(os.getcwd(), path, 'chip_model')
path_filename_trans = os.path.join(os.getcwd(), path, 'trans_model')

os.makedirs(path, exist_ok=True)
os.makedirs(path_filename_chip, exist_ok=True)
os.makedirs(path_filename_trans, exist_ok=True)

model = Model(mci_field, mci_obs, cat_pro)
Chip_list = []
loops = 2000
for loop in range(loops):
    model.cal_chip_cat_model()

    # ra_obs, dec_obs = model.chip_cel_obs()
    #
    # model.up_cat_epoch(ra_obs, dec_obs)
    #
    # model.cal_obs_pm()
    #
    # model.cal_chip_cat_model_new()
    Chip_list.append(model.chip_model_list)

# Chip_list = [[np.array([[1, 0, 0], [0, 1, 0]])] * 10] * loops
# ----------------------------------------------------------------------------------------------------------------------
set_seed(2)
m31_pm = M31_PM(mci_field, mci_obs)

path = 'M31_simulation/Star_scatters/Data'
filename = f'M31_disk_{field_index}.csv'
m31_pm.get_star_local(os.path.join(path, filename))

m31_pm.cal_star_coord()
m31_pm.cal_star_v()

m31_pm.cal_ideal_cat()
m31_pm.cal_ideal_pm()

pix_err_f_m31 = 0.15
res = [[] for _ in range(blocks_num ** 2)]
num_stars_part = []
for model_list_num in range(len(Chip_list)):
    # model_list_num = 0
    chip_model_list = Chip_list[model_list_num]

    m31_pm.cal_mag_err(pix_err_f_m31)

    m31_pm.cal_chip_cat()

    m31_pm.chip_cel_obs(chip_model_list)

    mid_num = int((len(chip_model_list) - 1) / 2)
    m31_pm.cal_obs_pm(mid_num)

    m31_pm.obs_pm_part(res)

columns = ['ra_pm_ideal', 'ra_pm_obs', 'dec_pm_ideal', 'dec_pm_obs']

path = 'M31_PM/PM_calculate/Data/PM_result_obs_val'
for part_index, res_part in zip(range(1, blocks_num ** 2 + 1), res):
    res_part = pd.DataFrame(res_part, columns=columns)
    filename = f'res_{field_index}_{obs_times}_{part_index}_val.csv'
    res_part.to_csv(os.path.join(path, filename), index=False)
