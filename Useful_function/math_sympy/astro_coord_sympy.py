import sympy as sym
from sympy import sin, cos
from sympy import diff
from sympy import sqrt
from sympy import Matrix, det, vector

from sympy import trigsimp, simplify
from sympy import Function


def matrix_x(angle):
    return Matrix(
        [[1, 0, 0], [0, cos(angle), sin(angle)], [0, -sin(angle), cos(angle)]]
    )


def matrix_y(angle):
    return Matrix(
        [[cos(angle), 0, -sin(angle)], [0, 1, 0], [sin(angle), 0, cos(angle)]]
    )


def matrix_z(angle):
    return Matrix(
        [[cos(angle), sin(angle), 0], [-sin(angle), cos(angle), 0], [0, 0, 1]]
    )


# -------------------------------------------------------------------------------
rho, phi = sym.symbols("rho, phi")
i, PA = sym.symbols("i, PA")

alpha, delta = sym.symbols("alpha, delta")
alpha_0, delta_0 = sym.symbols("alpha_0, delta_0")
Gamma = sym.symbols("Gamma")

# --------------------------------------------------------------------------------
# transformations between different coordinate systems
M_d2c = matrix_z(-PA) @ matrix_x(i)

M_r2l = Matrix([[0, 0, -1], [1, 0, 0], [0, 1, 0]])
M_c2f = M_r2l @ matrix_y(-rho) @ matrix_z(phi)

M_d2f = M_c2f @ M_d2c  # trigsimp(M_d2f)

cos_Gamma = (
    sin(delta) * cos(delta_0) * cos(alpha - alpha_0) - cos(delta) * sin(delta_0)
) / sin(rho)
sin_Gamma = cos(delta_0) * sin(alpha - alpha_0) / sin(rho)

# --------------------------------------------------------------------------------
#
D, D0 = sym.symbols("D, D0")
x, y, z = sym.symbols("x, y, z")

# z in disk plane is 0
D = D0 * cos(i) / (cos(i) * cos(rho) - sin(i) * sin(rho) * sin(phi - PA))

x = D * sin(rho) * cos(phi)
y = D * sin(rho) * sin(phi)
z = D0 - D * cos(rho)
P_c = Matrix([x, y, z])
P_d = M_d2c.inv() @ P_c
P_d = trigsimp(P_d)

# --------------------------------------------------------------------------------
# Velocity in different coordinate systems, and final velocity is in field.

# COM velocity
V_t, V_sys = sym.symbols("V_t, V_sys")
PA_t = sym.symbols("PA_t")

V_CM_c = Matrix([V_t * cos(PA_t), V_t * sin(PA_t), -V_sys])
V_CM_f = M_c2f @ V_CM_c

# velocity cause by Precession and Nutation
t = sym.symbols("t")

V_PN_c_PA = diff(M_d2c, PA) @ P_d * Function("PA")(t).diff(t)
V_PN_c_i = diff(M_d2c, i) @ P_d * Function("i")(t).diff(t)

V_PN_f_PA = trigsimp(M_c2f @ V_PN_c_PA)
V_PN_f_i = trigsimp(M_c2f @ V_PN_c_i)

# rotation velocity
sign = sym.symbols("sign")
R, V_R = sym.symbols("R, V_R")

# R = trigsimp(sqrt(P_d[0] ** 2 + P_d[1] ** 2))
R = (
    D0
    * sin(rho)
    * (-sin(i) ** 2 * cos(PA - phi) ** 2 + 1) ** (1 / 2)
    / (cos(i) * cos(rho) - sin(i) * sin(rho) * sin(phi - PA))
)

M_temp = Matrix([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
V_R_d = trigsimp(M_temp @ P_d * (sign * V_R / R))
V_R_f = trigsimp(M_d2f @ V_R_d)

V_f = V_CM_f + V_PN_f_PA + V_PN_f_i + V_R_f

# --------------------------------------------------------------------------------

V_f[0]

# np.sum((1 / dz**2) / np.sum(1 / dz**2) * z)
# temp = 129.9/180*np.pi
# (-402.9 / 50.1 / 4.7403885) + (-1.68*np.cos(temp) + 0.34*np.sin(temp))
