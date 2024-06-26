import numpy as np


def transform_matrix_x_f(angle):
    angle = np.deg2rad(angle)
    transform_matrix_x = np.array(
        [
            [1, 0, 0],
            [0, np.cos(angle), np.sin(angle)],
            [0, -np.sin(angle), np.cos(angle)],
        ]
    )
    return transform_matrix_x


def transform_matrix_y_f(angle):
    angle = np.deg2rad(angle)
    transform_matrix_y = np.array(
        [
            [np.cos(angle), 0, -np.sin(angle)],
            [0, 1, 0],
            [np.sin(angle), 0, np.cos(angle)],
        ]
    )
    return transform_matrix_y


def transform_matrix_z_f(angle):
    angle = np.deg2rad(angle)
    transform_matrix_z = np.array(
        [
            [np.cos(angle), np.sin(angle), 0],
            [-np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    return transform_matrix_z


def local_to_observed_coordinate_M31(position_xyz_loc, inclination, position_angle):
    """

    :param position_xyz_loc: the position of local coordinate of objects in the simulated M31 disk
    :param inclination: M31 inclination on the sky
    :param position_angle: M31 position_angle on the sky
    :param distance: the distance between observer and M31 center (to normalize local position value)
    :return: the position of observed coordinate of objects in the view
    """
    angle_1 = inclination
    angle_2 = -position_angle
    position_xyz_loc = position_xyz_loc

    # first transform
    transform_matrix_x = transform_matrix_x_f(angle_1)
    position_xyz_trans1 = np.dot(transform_matrix_x, position_xyz_loc)

    # second transform
    transform_matrix_z = transform_matrix_z_f(angle_2)
    position_xyz_trans2 = np.dot(transform_matrix_z, position_xyz_trans1)

    return position_xyz_trans2


def observed_to_starbased_coordinate(position_xyz_obs, distance):
    """

    :param position_xyz_obs:
    :return:
    """

    position_x_observed = position_xyz_obs[0, :]
    position_y_observed = position_xyz_obs[1, :]
    position_z_observed = position_xyz_obs[2, :]

    w = -position_z_observed
    xi = position_y_observed / (w + distance)
    eta = position_x_observed / (w + distance)
    w = w / w

    position_starbased_coordinate = np.array([xi, eta, w])
    return position_starbased_coordinate


def ideal_cel_coord(xi, eta, ra_c, dec_c):
    """
    transform objects' ideal coordinates to their celestial coordinates
    :param xi: ideal first coordinate, along the direction of ra increase
    :param eta: ideal second coordinate, along the direction of dec increase
    :param ra_c: ra of ideal plane tangent point (the central point of galaxy's ra)
    :param dec_c: dec of ideal plane tangent point (the central point of galaxy's dec)
    :return: ra, dec
    """
    ra_c = np.deg2rad(ra_c)
    dec_c = np.deg2rad(dec_c)

    ra = np.arctan(xi / (np.cos(dec_c) - eta * np.sin(dec_c))) + ra_c
    dec = np.arctan(
        (eta * np.cos(dec_c) + np.sin(dec_c))
        / (np.cos(dec_c) - eta * np.sin(dec_c))
        * np.cos(ra - ra_c)
    )
    ra = np.degrees(ra)
    dec = np.degrees(dec)

    return ra, dec


def cel_ideal_coord(ra, dec, ra_c, dec_c):
    """
    transform objects' celestial coordinates to their ideal coordinates
    :param ra: ra of objects in celestial coordinates
    :param dec: dec of objects in celestial coordinates
    :param ra_c: ra of ideal plane tangent point (the central point of galaxy's ra)
    :param dec_c: dec of ideal plane tangent point (the central point of galaxy's dec)
    :return: xi, eta
    """
    ra = np.radians(ra)
    dec = np.radians(dec)
    ra_c = np.radians(ra_c)
    dec_c = np.radians(dec_c)

    xi = (np.cos(dec) * np.sin(ra - ra_c)) / (
        np.sin(dec_c) * np.sin(dec) + np.cos(dec_c) * np.cos(dec) * np.cos(ra - ra_c)
    )
    eta = (
        np.cos(dec_c) * np.sin(dec) - np.sin(dec_c) * np.cos(dec) * np.cos(ra - ra_c)
    ) / (np.sin(dec_c) * np.sin(dec) + np.cos(dec_c) * np.cos(dec) * np.cos(ra - ra_c))

    return xi, eta


def starbased_to_starbased_coordinate(
    position_starbased_coordinate, ra_trans_c, dec_trans_c, ra_c, dec_c
):
    """

    :param position_starbased_coordinate:
    :param ra_c: the central ra of first star based coordinate, unit is degree
    :param dec_c: the central dec of first star based coordinate
    :param ra_trans_c: the central ra of transformed star based coordinate
    :param dec_trans_c: the central dec of transformed star based coordinate
    :return:
    """

    R_x_star1_to_icrs = transform_matrix_x_f(-90 + dec_c)
    R_z_star1_to_icrs = transform_matrix_z_f(-90 - ra_c)
    R_z_icrs_to_star2 = transform_matrix_z_f(90 + ra_trans_c)
    R_x_icrs_to_star2 = transform_matrix_x_f(90 - dec_trans_c)

    R_star1_to_star2 = (
        R_x_icrs_to_star2 @ R_z_icrs_to_star2 @ R_z_star1_to_icrs @ R_x_star1_to_icrs
    )

    position_starbased_coordinate_trans = (
        R_star1_to_star2 @ position_starbased_coordinate
    )
    project_normalize_factors = position_starbased_coordinate_trans[2, :]
    position_starbased_coordinate_trans = (
        position_starbased_coordinate_trans / project_normalize_factors
    )

    return position_starbased_coordinate_trans


def template_linear_interpolation(
    position_x, position_x_coord, value_correspond_x, value_correspond_x_coord
):
    for i, index in zip(position_x, range(len(position_x))):
        if i < position_x_coord[0]:
            value_correspond_x[index] = value_correspond_x_coord[0] + (
                i - position_x_coord[0]
            ) / (position_x_coord[1] - position_x_coord[0]) * (
                value_correspond_x_coord[1] - value_correspond_x_coord[0]
            )

        elif i > position_x_coord[-1]:
            value_correspond_x[index] = value_correspond_x_coord[-1] + (
                i - position_x_coord[-1]
            ) / (position_x_coord[-2] - position_x_coord[-1]) * (
                value_correspond_x_coord[-2] - value_correspond_x_coord[-1]
            )

        else:
            sub = position_x_coord - i
            lower_index = np.argmin(sub[0:-1] * sub[1:])
            upper_index = lower_index + 1

            value_correspond_x[index] = value_correspond_x_coord[lower_index] + (
                i - position_x_coord[lower_index]
            ) / (position_x_coord[upper_index] - position_x_coord[lower_index]) * (
                value_correspond_x_coord[upper_index]
                - value_correspond_x_coord[lower_index]
            )

    return value_correspond_x


def local_cel_ideal_coord(
    x_local,
    y_local,
    distance,
    inclination,
    position_angle,
    ra_m31_c,
    dec_m31_c,
    mci_field_ra_dec,
    MCI_field_size,
):
    position_xyz_loc = np.array(
        [np.array(x_local), np.array(y_local), np.zeros(len(x_local))]
    )

    position_xyz_observed = local_to_observed_coordinate_M31(
        position_xyz_loc, inclination, position_angle
    )
    position_starbased_coordinate_m31 = observed_to_starbased_coordinate(
        position_xyz_observed, distance
    )

    xi_m31 = position_starbased_coordinate_m31[0, :]
    eta_m31 = position_starbased_coordinate_m31[1, :]

    [ra_m31, dec_m31] = ideal_cel_coord(xi_m31, eta_m31, ra_m31_c, dec_m31_c)

    position_starbased_coordinate_disk = starbased_to_starbased_coordinate(
        position_starbased_coordinate_m31,
        mci_field_ra_dec[0],
        mci_field_ra_dec[1],
        ra_m31_c,
        dec_m31_c,
    )

    xi_disk = position_starbased_coordinate_disk[0, :]
    eta_disk = position_starbased_coordinate_disk[1, :]

    [ra_disk, dec_disk] = ideal_cel_coord(
        xi_disk, eta_disk, mci_field_ra_dec[0], mci_field_ra_dec[1]
    )

    MCI_field_size_rad = MCI_field_size / 2 / 180 * np.pi
    mci_field_index_flag = (np.abs(xi_disk) < MCI_field_size_rad) & (
        np.abs(eta_disk) < MCI_field_size_rad
    )
    # mci_field_index_flag = (xi_disk ** 2 + eta_disk ** 2) < (MCI_field_size / 2 / 180 * np.pi) ** 2

    xi_mci_field = xi_disk[mci_field_index_flag]
    eta_mci_field = eta_disk[mci_field_index_flag]

    ra_mci_field = ra_disk[mci_field_index_flag]
    dec_mci_field = dec_disk[mci_field_index_flag]

    return (
        xi_mci_field,
        eta_mci_field,
        ra_mci_field,
        dec_mci_field,
        mci_field_index_flag,
    )


def disk_rot_ang_v(disk_radius_local, rotation_curve_r, rotation_angle_v):
    """
    linear interpolation, disk_radius_local -> rotation_curve_r -> rotation_angle_v -> disk_angle_v
    to get the targeted scatters' angular velocities (disk_angle_v)
    :param disk_radius_local:
    :param rotation_curve_r:
    :param rotation_angle_v:
    :return: disk_angle_v
    """
    disk_radius = np.array(disk_radius_local)
    rotation_radius = np.array(rotation_curve_r)
    disk_angle_v = np.zeros(len(disk_radius))

    for i, index in zip(disk_radius, np.arange(len(disk_radius))):
        sub = rotation_radius - i
        lower_index = np.argmin(sub[0:-1] * sub[1:])
        upper_index = lower_index + 1

        disk_angle_v[index] = rotation_angle_v[lower_index] + (
            i - rotation_radius[lower_index]
        ) * (rotation_angle_v[upper_index] - rotation_angle_v[lower_index]) / (
            rotation_radius[upper_index] - rotation_radius[lower_index]
        )
    return disk_angle_v


def polynomial_least_square_(observed_data_x, observed_data_y, observed_data_x_num):
    observed_vector = np.zeros(observed_data_x_num)
    for i in range(observed_data_x_num):
        observed_vector[i] = np.sum(observed_data_x[:] * observed_data_y)

    coefficient_matrix = np.zeros((observed_data_x_num, observed_data_x_num))
    for i in range(observed_data_x_num):
        for j in range(observed_data_x_num):
            coefficient_matrix[i, j] = np.sum(observed_data_x[:] * observed_data_x[:])

    # solve linear equations
    fitting_coefficient = np.dot(np.linalg.inv(coefficient_matrix), observed_vector)
    # fitting_coefficient = np.linalg.inv(coefficient_matrix) @ vector
    return fitting_coefficient


def polynomial_least_square(observed_data_x, observed_data_y, observed_data_x_num):
    observed_vector = np.zeros((observed_data_x_num, 1))
    for i in range(observed_data_x_num):
        observed_vector[i] = np.sum(observed_data_x[i, :] * observed_data_y)

    coefficient_matrix = np.zeros((observed_data_x_num, observed_data_x_num))
    for i in range(observed_data_x_num):
        for j in range(observed_data_x_num):
            coefficient_matrix[i, j] = np.sum(
                observed_data_x[i, :] * observed_data_x[j, :]
            )

    # solve linear equations
    fitting_coefficient = np.dot(np.linalg.inv(coefficient_matrix), observed_vector)
    # fitting_coefficient = np.linalg.inv(coefficient_matrix) @ vector
    return fitting_coefficient
