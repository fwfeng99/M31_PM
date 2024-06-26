import datetime
import random
import numpy as np
import pandas as pd
import emcee
from scipy.optimize import minimize

from astropy import units as u
from astropy import constants as const
from astropy.coordinates import SkyCoord
from astropy.time import Time

import matplotlib.pyplot as plt

import os
import sys
import glob
import re

sys.path.extend(['D:\\repos\\PycharmProjects\\M31_kinematics_validate', 'D:/repos/PycharmProjects/M31_kinematics_validate'])
import function


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
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
        self.M31_c = SkyCoord.from_name('M31')
        self.ra_list = field_ra_list
        self.dec_list = field_dec_list
        self.index = field_index

        self.coord = np.array([field_ra_list[field_index - 1], field_dec_list[field_index - 1]])
        self.coord_ideal = function.cel_ideal_coord(self.coord[0], self.coord[1], self.M31_c.ra.value, self.M31_c.dec.value)
        self.rho = np.arctan((self.coord_ideal[0] ** 2 + self.coord_ideal[1] ** 2) ** 0.5)
        self.phi = np.pi - np.arctan2(self.coord_ideal[1], self.coord_ideal[0])

        self.blocks = blocks_num
        blocks_index = np.linspace(-self.blocks + 1, self.blocks - 1, self.blocks)
        blocks_index_list = [[j, i] for i in blocks_index for j in blocks_index]

        mci_field_size_part = self.mci_field_size / self.blocks * (np.pi / 180)
        self.coord_part = [function.ideal_cel_coord(x * mci_field_size_part / 2, y * mci_field_size_part / 2,
                                                    self.coord[0], self.coord[1]) for x, y in blocks_index_list]

        self.coord_ideal_part = function.cel_ideal_coord(np.array(self.coord_part)[:, 0], np.array(self.coord_part)[:, 1],
                                                         self.M31_c.ra.value, self.M31_c.dec.value)

        self.rho_part = np.arctan((self.coord_ideal_part[0] ** 2 + self.coord_ideal_part[1] ** 2) ** 0.5)
        self.phi_part = np.pi - np.arctan2(self.coord_ideal_part[1], self.coord_ideal_part[0])

        self.M31_dis = 785  # kpc
        self.M31_inc = 73.7
        self.M31_PA = 38.3
        self.v_sys = -301


class PM_COM_fit(object):

    def __init__(self, mci_field):
        self.field = mci_field

    def v_para_factor(self, *args):
        if len(args) == 1:
            rho = self.field.rho_part[args]
            phi = self.field.phi_part[args]
            p_t = self.field.phi_part[args] - self.field.M31_PA * np.pi / 180 - np.pi / 2  # phi substracts theta(PA)
        else:
            rho = self.field.rho
            phi = self.field.phi
            p_t = self.field.phi - self.field.M31_PA * np.pi / 180 - np.pi / 2  # phi substracts theta(PA)

        i = self.field.M31_inc * np.pi / 180
        v_sys = self.field.v_sys

        x1 = np.cos(rho) * np.cos(phi)
        x2 = np.cos(rho) * np.sin(phi)
        x3 = -v_sys * np.sin(rho)
        x4 = np.sin(i) * np.cos(p_t) * (np.cos(i) * np.sin(rho) + np.sin(i) * np.cos(rho) * np.sin(p_t)) / (
                np.cos(i) ** 2 * np.cos(p_t) ** 2 + np.sin(p_t) ** 2) ** 0.5

        x5 = -np.sin(phi)
        x6 = np.cos(phi)
        x7 = -(np.cos(i) ** 2 * np.cos(p_t) ** 2 + np.sin(p_t) ** 2) / (np.cos(i) ** 2 * np.cos(p_t) ** 2 + np.sin(p_t) ** 2) ** 0.5

        return [x1, x2, x3, x4, x5, x6, x7]

    def u_v(self, *args):
        if (len(args) == 1):
            rho = self.field.rho_part[args]
            p_t = self.field.phi_part[args] - self.field.M31_PA * np.pi / 180 - np.pi / 2  # phi substracts theta(PA)
            ra = np.deg2rad(self.field.coord[0])
            dec = np.deg2rad(self.field.coord[1])
        else:
            rho = self.field.rho
            p_t = self.field.phi - self.field.M31_PA * np.pi / 180 - np.pi / 2  # phi substracts theta(PA)
            ra = np.deg2rad(self.field.coord[0])
            dec = np.deg2rad(self.field.coord[1])

        i = self.field.M31_inc * np.pi / 180
        D0 = self.field.M31_dis
        ra_c = self.field.M31_c.ra.rad
        dec_c = self.field.M31_c.dec.rad

        cos_T = (np.sin(dec) * np.cos(dec_c) * np.cos(ra - ra_c) - np.cos(dec) * np.sin(dec_c)) / np.sin(rho)
        sin_T = np.cos(dec_c) * np.sin(ra - ra_c) / np.sin(rho)
        D_f = (np.cos(i) * np.cos(rho) - np.sin(i) * np.sin(rho) * np.sin(p_t)) / (D0 * np.cos(i))
        rot_f = np.array([[-sin_T, -cos_T], [cos_T, -sin_T]])

        R_uv = 1 / D_f * np.linalg.inv(rot_f) * 4.740470463533348
        # v_23 = 1 / D_f * np.linalg.inv(rot_f) @ np.array([u_w, u_n]) * 4.740470463533348

        return R_uv


class MCMC_fit(object):
    def __init__(self, x1, x2, x3, x4, x5, x6, y1, y2, sigma1, sigma2):
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3
        self.x4 = x4
        self.x5 = x5
        self.x6 = x6
        self.y1 = y1
        self.y2 = y2
        self.sigma1 = sigma1
        self.sigma2 = sigma2

    def ln_prior(self, params):
        a, b = params
        if -200 < a < 300 and -2 * np.pi < b < 2 * np.pi:
            return 0.0  # 先验概率为常数
        return -np.inf  # 参数超出先验范围时的先验概率为零

    # 定义似然函数
    def ln_likelihood(self, params):
        a, b = params
        model_y1 = self.x1 * a * np.cos(b) + self.x2 * a * np.sin(b) + self.x3
        model_y2 = self.x4 * a * np.cos(b) + self.x5 * a * np.sin(b) + self.x6
        ln_likelihood_y1 = -0.5 * np.sum((y1 - model_y1) ** 2 / self.sigma1 ** 2)
        ln_likelihood_y2 = -0.5 * np.sum((y2 - model_y2) ** 2 / self.sigma2 ** 2)
        return ln_likelihood_y1 + ln_likelihood_y2

    # 定义总的对数概率函数
    def ln_probability(self, params):
        lp = self.ln_prior(params)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.ln_likelihood(params)

    def run_mcmc(self):
        # 设置 MCMC 链的参数
        ndim = 2  # 参数的数量
        nwalkers = 200  # 随机行走者的数量
        nsteps = 3000  # 每个行走者的步数

        # 初始化行走者的起始位置
        initial_params = np.array([100, 0])  # 初始参数估计
        initial_pos = initial_params + 1e-3 * np.random.randn(nwalkers, ndim)

        # 创建 MCMC 采样器
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.ln_probability, args=())

        # 运行 MCMC 采样
        sampler.run_mcmc(initial_pos, nsteps)

        # 获取采样结果
        samples = sampler.chain  # 采样轨迹
        samples = samples[:, :, :].reshape((-1, ndim))  # 展平

        return samples


def error_function(params, x1, x2, x3, x4, x5, x6, x7, y1_obs, y2_obs):
    y1_model, y2_model = model(params, x1, x2, x3, x4, x5, x6, x7)
    residuals_y1 = y1_model - y1_obs
    residuals_y2 = y2_model - y2_obs
    total_residuals = np.concatenate((residuals_y1, residuals_y2))
    return np.sum(total_residuals ** 2)


def model(params, x1, x2, x3, x4, x5, x6, x7):
    a, b, c = params
    y1 = x1 * a * np.cos(b) + x2 * a * np.sin(b) + x3 + x4 * c
    y2 = x5 * a * np.cos(b) + x6 * a * np.sin(b) + x7 * c
    return y1, y2


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
set_seed(1)

field_ra_list = np.array([12.28574129, 11.22870306, 9.15566471, 10.09915259, 11.25])
field_dec_list = np.array([42.75043505, 40.95362697, 39.76631034, 41.6017492, 42.48])
obs_times = 10
field_index_start = 1
field_index_end = 1
blocks_num = 2

path = 'M31_PM/PM_calculate/Data/PM_result_obs'
x = []
y = []
R_uv = []

for field_index in range(field_index_start, field_index_end + 1):
    # field_index = 1
    mci_field = MCI_Field(field_ra_list, field_dec_list, field_index, blocks_num)

    PM_COM = PM_COM_fit(mci_field)

    # 观测数据和自变量
    files = glob.glob(os.path.join(path, f"*{field_index}*{obs_times}_*.csv"))
    for file in files:
        # print(file)
        pm_data = pd.read_csv(file)
        pm_ra_obs = np.array(pm_data['ra_pm_obs'])
        pm_dec_obs = np.array(pm_data['dec_pm_obs'])

        field_part_index = int(re.search(r'\d+', file[::-1]).group()[::-1])

        pm_w = -pm_ra_obs * np.cos(np.deg2rad(mci_field.coord_part[field_part_index - 1][1])) / 1000
        pm_n = pm_dec_obs / 1000

        for i in range(len(pm_w)):
            vp_temp = PM_COM.v_para_factor()
            R_uv_temp = PM_COM.u_v()
            x.append(vp_temp)
            R_uv.append(R_uv_temp)
            y.append(R_uv_temp @ np.array([pm_w[i], pm_n[i]]))

# ------------------------------------------------------------------------------------------------------------------------------------------
x = np.array(x)
y = np.array(y)
[a, b, c] = [100, -1, -210]
params = []
sta_times = len(pm_w)

for i in range(sta_times):
    [y1, y2] = [y[i::sta_times, j] for j in range(2)]
    [x1, x2, x3, x4, x5, x6, x7] = [x[i::sta_times, j] for j in range(7)]

    initial_guess = [np.random.normal(a, 100), np.random.normal(b, 1), np.random.normal(c, 20)]  # 请提供适当的初始值

    result = minimize(error_function, initial_guess, args=(x1, x2, x3, x4, x5, x6, x7, y1, y2))
    params.append(result.x)

res = np.array(params)

# ------------------------------------------------------------------------------------------------------------------------------------------

[vt, theta, Vr] = [res[:, i] for i in range(3)]
index = (vt > 0) & (theta < 0) & (Vr < -100)
[vt, theta, Vr] = [vt[index], theta[index], Vr[index]]
print(np.mean(vt), np.std(vt))
print(np.mean(theta), np.std(theta))
print(np.mean(Vr), np.std(Vr))

sta_pm = []

for field_index in range(field_index_start, field_index_end + 1):
    mci_field = MCI_Field(field_ra_list, field_dec_list, field_index, blocks_num)

    PM_COM = PM_COM_fit(mci_field)

    files = glob.glob(os.path.join(path, f"*{field_index}*{obs_times}*.csv"))
    for file in files:
        pm_data = pd.read_csv(file)
        pm_ra_ideal = np.array(pm_data['ra_pm_ideal'])
        pm_dec_ideal = np.array(pm_data['dec_pm_ideal'])

        field_part_index = int(re.search(r'\d+', file[::-1]).group()[::-1])

        pm_w = -pm_ra_ideal * np.cos(np.deg2rad(mci_field.coord_part[field_part_index - 1][1])) / 1000
        pm_n = pm_dec_ideal / 1000
        pm_w = pm_w[index]
        pm_n = pm_n[index]

        vp_temp = PM_COM.v_para_factor(field_part_index - 1)
        R_uv_temp = PM_COM.u_v(field_part_index - 1)
        # vp_temp = PM_COM.v_para_factor()
        # R_uv_temp = PM_COM.u_v()
        y1_temp = vp_temp[0] * vt * np.cos(theta) + vp_temp[1] * vt * np.sin(theta) + vp_temp[2] + vp_temp[3] * Vr
        y2_temp = vp_temp[4] * vt * np.cos(theta) + vp_temp[5] * vt * np.sin(theta) + vp_temp[6] * Vr
        pm_wn_temp = np.linalg.inv(R_uv_temp) @ np.array([y1_temp, y2_temp]) * 1000

        sta_pm.append([np.mean(pm_w) * 1000, np.mean(pm_wn_temp, axis=1)[0], np.std(pm_wn_temp, axis=1)[0],
                       np.mean(pm_n) * 1000, np.mean(pm_wn_temp, axis=1)[1], np.std(pm_wn_temp, axis=1)[1]])

sta_pm = np.array(sta_pm)

# ------------------------------------------------------------------------------------------------------------------------------------------

# fig = plt.figure(figsize=(5, 5))
# ax = fig.add_subplot(1, 1, 1)
# plt.hist(vt, bins=70,
#          color='blue', alpha=0.5, edgecolor='black', linewidth=1,
#          label='Vt')
#
# xlabel_text = r'$V_t$ [$Km/s$]'
# ax.set_xlabel(xlabel_text, fontweight='bold', fontsize=14)
# ax.set_ylabel('Numbers', fontweight='bold', fontsize=14)
# ax.tick_params(axis='x', labelsize=12, width=2)
# ax.tick_params(axis='y', labelsize=12, width=2)
# ax.text(0.1, 0.9, '(' + 'a' + ')', transform=ax.transAxes, fontsize=12, fontweight='bold')

# for field_index in range(field_index_start, field_index_end + 1):
#     mci_field = MCI_Field(field_ra_list, field_dec_list, field_index)
#
#     PM_COM = PM_COM_fit(mci_field)
#
#     path = 'M31_PM/PM_calculate/Data/PM_result_obs'
#     files = glob.glob(os.path.join(path, f"*{field_index}*{obs_times}*.csv"))
#     for file in files:
#         pm_data = pd.read_csv(file)
#         pm_ra_ideal = np.array(pm_data['ra_pm_ideal'])
#         pm_dec_ideal = np.array(pm_data['dec_pm_ideal'])
#         pm_ra_obs = np.array(pm_data['ra_pm_obs'])
#         pm_dec_obs = np.array(pm_data['dec_pm_obs'])
#
#         print(np.mean(pm_ra_ideal) * 0.734, np.mean(pm_ra_obs) * 0.734, np.std(pm_ra_ideal) * 0.734, np.std(pm_ra_obs) * 0.734)
#         print(np.mean(pm_dec_ideal), np.mean(pm_dec_obs), np.std(pm_dec_ideal), np.std(pm_dec_obs))
