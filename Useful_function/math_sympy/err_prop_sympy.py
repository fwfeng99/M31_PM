import sympy as sym
from sympy import sin, cos, exp, sqrt
from sympy import Matrix, det
from sympy import diff
from sympy import simplify

# ------------------------------------------------------------------------------------------------------------------------------------------
# define symbols
x1, x2 = sym.symbols("r, Mr")
dx1, dx2 = sym.symbols("dr, dMr")
y = sym.symbols("d")

# ------------------------------------------------------------------------------------------------------------------------------------------
# equation for error propagation calculation
equation = 0.01 * 10 ** (0.2 * (x1 - x2))

# ------------------------------------------------------------------------------------------------------------------------------------------
# the result
deltay = sqrt((diff(equation, x1) * dx1) ** 2 + (diff(equation, x2) * dx2) ** 2)
# simplify(deltay)
print(sym.latex(deltay))

# ------------------------------------------------------------------------------------------------------------------------------------------
# value assignment
values = {x1: 3.0, x2: 1.0, dx1: 0.1, dx2: 0.2}
deltay_value = deltay.subs(values).evalf()
