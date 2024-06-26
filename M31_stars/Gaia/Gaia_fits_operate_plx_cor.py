import numpy as np
import pandas as pd

from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import units as u
import os
import sys

import function

sys.path.extend(['D:/Code/Python/M31_Kinematics/M31_stars/Gaia'])
import parallax_cor as para_cor
from parallax_cor import *

# read star catalogue

path = 'M31_stars/Gaia/Data/'
# filename = 'Gaiadr3.fit'
filename = 'Gaiadr3_M31.fit'
path_filename = os.path.join(path, filename)

hdu = fits.open(path_filename)
print('文件信息：')
hdu.info()

# print(repr(hdu[1]))
# print(repr(hdu[1].header[:]))

region_data = Table.read(path_filename, hdu=1).to_pandas()

ra = region_data['RA_ICRS']
dec = region_data['DE_ICRS']
ra_error = region_data['e_RA_ICRS']  # unit is mas
dec_error = region_data['e_DE_ICRS']
plx = region_data['Plx']
plx_error = region_data['e_Plx']
PM = region_data['PM']  # unit is mas / year
pm_ra = region_data['pmRA']
pm_dec = region_data['pmDE']
pm_ra_error = region_data['e_pmRA']
pm_dec_error = region_data['e_pmDE']
gmag = region_data['Gmag']
nueff = region_data['nueff']
pscol = region_data['pscol']

# / np.cos(dec * np.pi / 180)

# mci view field information

# disk = SkyCoord('00 49 08.6 +42 45 02', unit=(u.hourangle, u.deg), obstime=Time('J2000'))
# mci_disk_ra_dec = [disk.ra.degree, disk.dec.degree]

area_index = 8
disk_ra_list = np.array([12.28574129, 11.22870306, 9.15566471, 10.09915259, 12.064, 10.331, 9.185, 11.250])
disk_dec_list = np.array([42.75043505, 40.95362697, 39.76631034, 41.6017492, 41.865, 40.195, 40.435, 42.480])
mci_disk_ra_dec = np.array([disk_ra_list[area_index - 1], disk_dec_list[area_index - 1]])

mci_field_size = 0.128  # unit is degree
focal_length = 41253  # unit is mm
pixel_size = 0.01  # unit is mm
pixel_resolution = 50  # unit is mas

M31_c = SkyCoord.from_name('M31')

# ------------------------------------------------------------------------------------------------------------------------------------------
# ra_dec to xi_eta and data clean

[xi, eta] = function.cel_ideal_coord(ra, dec, mci_disk_ra_dec[0], mci_disk_ra_dec[1])

mci_field_size_rad = mci_field_size / 2 / 180 * np.pi
mci_field_index_flag = (np.abs(xi) < mci_field_size_rad) & (np.abs(eta) < mci_field_size_rad)

xi_mci_field = np.array(xi[mci_field_index_flag])
eta_mci_field = np.array(eta[mci_field_index_flag])
ra_mci_field = np.array(ra[mci_field_index_flag])
dec_mci_field = np.array(dec[mci_field_index_flag])
ra_mci_field_error = np.array(ra_error[mci_field_index_flag])
dec_mci_field_error = np.array(dec_error[mci_field_index_flag])
plx_mci_field = np.array(plx[mci_field_index_flag])
plx_mci_field_error = np.array(plx_error[mci_field_index_flag])
PM_mci_field = np.array(PM[mci_field_index_flag])
pm_ra_mci_field = np.array(pm_ra[mci_field_index_flag])
pm_dec_mci_field = np.array(pm_dec[mci_field_index_flag])
pm_ra_mci_field_error = np.array(pm_ra_error[mci_field_index_flag])
pm_dec_mci_field_error = np.array(pm_dec_error[mci_field_index_flag])
gmag_mci_field = np.array(gmag[mci_field_index_flag])
nueff_mci_field = np.array(nueff[mci_field_index_flag])
pscol_mci_field = np.array(pscol[mci_field_index_flag])

# data clean

clean_index_1 = ~(np.abs((gmag_mci_field - np.mean(gmag_mci_field)) / np.std(gmag_mci_field)) > 3)
clean_index_2 = ~(np.abs((ra_mci_field_error - np.mean(ra_mci_field_error)) / np.std(ra_mci_field_error)) > 3)
clean_index_3 = ~(np.abs((dec_mci_field_error - np.mean(dec_mci_field_error)) / np.std(dec_mci_field_error)) > 3)
clean_index_4 = (np.abs(pm_ra_mci_field) > 1e-7) & (np.abs(pm_dec_mci_field) > 1e-7) & (pm_ra_mci_field_error > 1e-7) & (
        pm_dec_mci_field_error > 1e-7)
clean_index_5 = (plx_mci_field > 0.0) & ~(np.abs((plx_mci_field_error - np.mean(plx_mci_field_error)) / np.std(plx_mci_field_error)) > 3)

clean_index = clean_index_1 & clean_index_2 & clean_index_3 & clean_index_4 & clean_index_5

xi_mci_field = xi_mci_field[clean_index]
eta_mci_field = eta_mci_field[clean_index]
ra_mci_field = ra_mci_field[clean_index]
dec_mci_field = dec_mci_field[clean_index]
ra_mci_field_error = ra_mci_field_error[clean_index]
dec_mci_field_error = dec_mci_field_error[clean_index]
plx_mci_field = plx_mci_field[clean_index]
plx_mci_field_error = plx_mci_field_error[clean_index]
PM_mci_field = PM_mci_field[clean_index]
pm_ra_mci_field = pm_ra_mci_field[clean_index]
pm_dec_mci_field = pm_dec_mci_field[clean_index]
pm_ra_mci_field_error = pm_ra_mci_field_error[clean_index]
pm_dec_mci_field_error = pm_dec_mci_field_error[clean_index]
gmag_mci_field = gmag_mci_field[clean_index]
nueff_mci_field = nueff_mci_field[clean_index]
pscol_mci_field = pscol_mci_field[clean_index]
flag_56 = nueff_mci_field > pscol_mci_field

coords = SkyCoord(
    ra=ra_mci_field * u.deg,
    dec=dec_mci_field * u.deg
)
beta = np.array([coords[i].barycentricmeanecliptic.lat.radian for i in range(len(flag_56))])
G = gmag_mci_field

Z = []

for i in range(len(flag_56)):
    if flag_56[i]:
        flag = 5
        veff = nueff_mci_field[i]
    else:
        flag = 6
        veff = pscol_mci_field[i]
    z = 0
    for j in range(5):
        for k in range(3):
            z += para_cor.Q_cor(j, k, G[i], flag) * para_cor.c_cor(j, veff) * para_cor.b_cor(k, beta[i])

    Z.append(z / 1000)

plx_cor_mci_field = plx_mci_field + np.array(Z)

clean_index = [plx_cor_mci_field > 0]


columns = ['ra', 'dec', 'e_ra', 'e_dec', 'plx', 'plx_cor', 'e_plx', 'pm_ra', 'pm_dec', 'e_pm_ra', 'e_pm_dec', 'gmag', 'nueff', 'pscol', 'flag_56']
data = pd.DataFrame(
    {'ra': ra_mci_field, 'dec': dec_mci_field, 'e_ra': ra_mci_field_error, 'e_dec': dec_mci_field_error,
     'plx': plx_mci_field, 'plx_cor': plx_cor_mci_field, 'e_plx': plx_mci_field_error, 'pm_ra': pm_ra_mci_field,
     'pm_dec': pm_dec_mci_field, 'e_pm_ra': pm_ra_mci_field_error, 'e_pm_dec': pm_dec_mci_field_error,
     'gmag': gmag_mci_field, 'nueff': nueff_mci_field, 'pecol': pscol_mci_field, 'flag_56': flag_56})


data = data[data['plx_cor'] > 0]

path = 'M31_stars/Gaia/Data/'
# data.to_csv(os.path.join(path, f'Gaia_M31_disk_{area_index}.csv'), index=False)
data.to_csv(os.path.join(path, f'Gaia_M31_disk_{area_index}_test.csv'), index=False)
