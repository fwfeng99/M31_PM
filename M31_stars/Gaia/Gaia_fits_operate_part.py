import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from astropy.io import fits
from astropy.table import Table
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

import sys

sys.path.extend(['D:\\Code\\Python\\M31_Kinematics', 'D:/Code/Python/M31_Kinematics'])
import function

# read star catalogue

in_path = 'M31_stars/Gaia/Data/'
filename = 'Gaiadr3.fit'
in_path_filename = in_path + filename

hdu = fits.open(in_path_filename)
print('文件信息：')
hdu.info()

print(repr(hdu[1]))
print(repr(hdu[1].header[:120]))

region_data = Table.read(in_path_filename, hdu=1).to_pandas()

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

# / np.cos(dec * np.pi / 180)

# mci view field information

disk = SkyCoord('00 49 08.6 +42 45 02', unit=(u.hourangle, u.deg), obstime=Time('J2000'))

mci_disk_ra_dec = [disk.ra.degree, disk.dec.degree]
mci_field_size = 0.128  # unit is degree
mci_field_size_part = mci_field_size / 2

focal_length = 41253  # unit is mm
pixel_size = 0.01  # unit is mm
pixel_resolution = 50  # unit is mas

M31_c = SkyCoord.from_name('M31')

mci_field_size_part = mci_field_size / 2
mci_field_size_part_rad = mci_field_size_part * (np.pi / 180)

mci_disk_ra_dec_list = [
    function.ideal_cel_coord(x * mci_field_size_part_rad / 2, y * mci_field_size_part_rad / 2, mci_disk_ra_dec[0],
                                           mci_disk_ra_dec[1]) for x, y in [(-1, -1), (1, -1), (-1, 1), (1, 1)]]

# ------------------------------------------------------------------------------------------------------------------------------------------
# ra_dec to xi_eta and data clean

for i, mci_disk_ra_dec in zip(range(1, 5), mci_disk_ra_dec_list):
    [xi, eta] = function.celestail_to_ideal_coordiante(ra, dec, mci_disk_ra_dec[0], mci_disk_ra_dec[1])

    mci_field_size_rad = mci_field_size_part / 2 / 180 * np.pi
    # mci_field_size_rad = mci_field_size / 2 / 180 * np.pi
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

    # data clean

    clean_index_1 = ~(np.abs((gmag_mci_field - np.mean(gmag_mci_field)) / np.std(gmag_mci_field)) > 3)
    clean_index_2 = ~(np.abs((ra_mci_field_error - np.mean(ra_mci_field_error)) / np.std(ra_mci_field_error)) > 3)
    clean_index_3 = ~(np.abs((dec_mci_field_error - np.mean(dec_mci_field_error)) / np.std(dec_mci_field_error)) > 3)
    clean_index_4 = (np.abs(pm_ra_mci_field) > 1e-7) & (np.abs(pm_dec_mci_field) > 1e-7) & (pm_ra_mci_field_error > 1e-7) & (
            pm_dec_mci_field_error > 1e-7)
    clean_index_5 = (plx_mci_field > 0.0) & ~(
                np.abs((plx_mci_field_error - np.mean(plx_mci_field_error)) / np.std(plx_mci_field_error)) > 3)

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

    columns = ['ra', 'dec', 'e_ra', 'e_dec', 'plx', 'e_plx', 'pm_ra', 'pm_dec', 'e_pm_ra', 'e_pm_dec', 'gmag']
    data = pd.DataFrame(
        {'ra': ra_mci_field, 'dec': dec_mci_field, 'e_ra': ra_mci_field_error, 'e_dec': dec_mci_field_error,
         'plx': plx_mci_field, 'e_plx': plx_mci_field_error, 'pm_ra': pm_ra_mci_field, 'pm_dec': pm_dec_mci_field,
         'e_pm_ra': pm_ra_mci_field_error, 'e_pm_dec': pm_dec_mci_field_error, 'gmag': gmag_mci_field})

    data.to_csv(in_path + 'Gaia_M31_disk' + '_part_' + str(i) + '.csv', index=False)

# plot validate
data_1 = pd.read_csv(in_path + 'Gaia_M31_disk' + '_part_' + str(1) + '.csv')
data_2 = pd.read_csv(in_path + 'Gaia_M31_disk' + '_part_' + str(2) + '.csv')
data_3 = pd.read_csv(in_path + 'Gaia_M31_disk' + '_part_' + str(3) + '.csv')
data_4 = pd.read_csv(in_path + 'Gaia_M31_disk' + '_part_' + str(4) + '.csv')

plt.plot(data_1['ra'], data_1['dec'], '.', markersize=5, c='r')
plt.plot(data_2['ra'], data_2['dec'], '.', markersize=5, c='b')
plt.plot(data_3['ra'], data_3['dec'], '.', markersize=5, c='y')
plt.plot(data_4['ra'], data_4['dec'], '.', markersize=5, c='g')
