import numpy as np
import pandas as pd
from scipy.stats import zscore

from astropy.io import fits
from astropy.table import Table
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

import os
import sys

sys.path.extend(['D:\\Code\\Python\\M31_Kinematics', 'D:/Code/Python/M31_Kinematics'])
import function_class


class MCI(object):
    mci_field_size = 0.128  # unit is degree
    focal_length = 41253  # unit is mm
    pixel_size = 0.01  # unit is mm
    pixel_resolution = 50  # unit is mas

    def __init__(self):
        pass


class MCI_Field(MCI):

    def __init__(self, ra_point, dec_point):
        self.ra_point = ra_point
        self.dec_point = dec_point


class Gaia_Fits(object):

    def __init__(self, path: str, filename: str):
        self.path = path
        self.filename = filename

    def open_fits(self):
        full_path = os.path.join(self.path, self.filename)
        # hdu = fits.open(full_path)
        self.data = Table.read(full_path, hdu=1).to_pandas()

    def get_columns(self, para_list: list):
        self.data_sel = self.data.loc[:, para_list]

    def filter_data(self, ra_field: float, dec_field: float, field_size: float):
        [self.data_sel['xi'], self.data_sel['eta']] = function_class.cel_ideal_coord(self.data_sel.loc[:, 'RA_ICRS'],
                                                                                     self.data_sel.loc[:, 'DE_ICRS'],
                                                                                     ra_field, dec_field)
        # mci field filter
        field_size_rad = field_size / 2 / 180 * np.pi
        self.data_sel = self.data_sel[(np.abs(self.data_sel['xi']) < field_size_rad) & (np.abs(self.data_sel['eta']) < field_size_rad)]

        # data quality filter
        # self.data_sel = self.data_sel[~(np.abs(zscore(self.data_sel['Gmag'])) > 3)]
        self.data_sel = self.data_sel[~(np.abs(zscore(self.data_sel['e_RA_ICRS'])) > 3)]
        self.data_sel = self.data_sel[~(np.abs(zscore(self.data_sel['e_DE_ICRS'])) > 3)]
        self.data_sel = self.data_sel[(np.abs(self.data_sel['pmRA']) > 1e-7) & (np.abs(self.data_sel['pmDE']) > 1e-7)
                                      & (self.data_sel['e_pmRA'] > 1e-7) & (self.data_sel['e_pmDE'] > 1e-7)]
        self.data_sel = self.data_sel[(np.abs(self.data_sel['Plx']) > 0.0) & ~(np.abs(zscore(self.data_sel['e_Plx'])) > 3)]


#
M31_c = SkyCoord.from_name('M31')
disk = SkyCoord('00 49 08.6 +42 45 02', unit=(u.hourangle, u.deg), obstime=Time('J2000'))
# ra_mci_field, dec_mci_field = disk.ra.degree, disk.dec.degree

mci = MCI()
mci_field = MCI_Field(disk.ra.degree, disk.dec.degree)

#
gaia_fits = Gaia_Fits('M31_stars/Gaia/Data/', 'Gaiadr3.fit')
gaia_fits.open_fits()

para_list = ['RA_ICRS', 'DE_ICRS', 'e_RA_ICRS', 'e_DE_ICRS', 'Plx', 'e_Plx', 'PM', 'pmRA', 'pmDE', 'e_pmRA', 'e_pmDE', 'Gmag']
gaia_fits.get_columns(para_list)
gaia_fits.filter_data(mci_field.ra_point, mci_field.dec_point, mci.mci_field_size)

# gaia_fits.data_sel[para_list].to_csv('Gaia_M31_disk_class.csv', index=False)
