import numpy as np
from scipy.optimize import minimize

# 合并观测数据和自变量
# x1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # 用自己的数据替换
# x2 = np.array([0.5, 1.0, 1.5, 2.0, 2.5])  # 用自己的数据替换
# x3 = np.array([0.2, 0.3, 0.4, 0.5, 0.6])  # 用自己的数据替换
# x4 = np.array([0.1, 0.2, 0.3, 0.4, 0.5])  # 用自己的数据替换
# y1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # 用自己的数据替换
# y2 = np.array([0.5, 1.0, 1.5, 2.0, 2.5])  # 用自己的数据替换
# sigma1 = np.array([0.1, 0.2, 0.1, 0.2, 0.1])  # y1的误差标准差
# sigma2 = np.array([0.2, 0.1, 0.2, 0.1, 0.2])  # y2的误差标准差

# 定义卡方损失函数
def chi_squared(params, x1, x2, x3, x4, y1, y2, sigma1, sigma2):
    a, b, c = params
    residuals_y1 = (y1 - (a * x1 * np.cos(b) + x2 * c + x3)) / sigma1
    residuals_y2 = (y2 + (a * x4 * np.sin(b) - x4 * c)) / sigma2
    chi_squared_y1 = np.sum(residuals_y1**2)
    chi_squared_y2 = np.sum(residuals_y2**2)
    return chi_squared_y1 + chi_squared_y2

# 初始参数估计
initial_params = np.array([100, np.pi / 2, 150])  # 初始参数估计

# 最小化卡方损失函数
result = minimize(chi_squared, initial_params, args=(x1, x2, x3, x4, y1, y2, sigma1, sigma2))
best_fit_params = result.x

# 输出估计的参数
a, b, c = best_fit_params
print(f"a = {a:.3f}")
print(f"b = {b:.3f}")
print(f"c = {c:.3f}")
