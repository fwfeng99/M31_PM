from scipy.optimize import minimize
import numpy as np

def error_function(params, x1, x2, x3, x4, x5, x6, y1_obs, y2_obs):
    y1_model, y2_model = model(params, x1, x2, x3, x4, x5, x6)
    residuals_y1 = y1_model - y1_obs
    residuals_y2 = y2_model - y2_obs
    total_residuals = np.concatenate((residuals_y1, residuals_y2))
    return np.sum(total_residuals ** 2)


def model(params, x1, x2, x3, x4, x5, x6):
    a, b = params
    y1 = x1 * a * np.cos(b) + x2 * a * np.sin(b) + x3
    y2 = x4 * a * np.cos(b) + x5 * a * np.sin(b) + x6
    return y1, y2

# np.random.seed(1)
x1 = np.random.uniform(0.74, 0.76, 1000)
x2 = np.random.uniform(0.59, 0.61, 1000)
x3 = np.random.uniform(-0.39, -0.41, 1000)
x4 = np.random.uniform(-0.6, -0.62, 1000)
x5 = np.random.uniform(0.77, 0.78, 1000)
x6 = np.random.uniform(-66, -68, 1000)

a = 400
b = -2

y1_obs = x1 * a * np.cos(b) + x2 * a * np.sin(b) + x3 + np.random.uniform(-10, 10, 1000)
y2_obs = x4 * a * np.cos(b) + x5 * a * np.sin(b) + x6 + np.random.uniform(-10, 10, 1000)

# 初始猜测参数值
initial_guess = [np.random.normal(200, 100), np.random.normal(b, 3)]  # 请提供适当的初始值

# 最小化误差函数，找到最优参数
result = minimize(error_function, initial_guess, args=(x1, x2, x3, x4, x5, x6, y1_obs, y2_obs))
# result = minimize(error_function, initial_guess, args=(x1, x2, x3, x4, x5, x6, y1, y2))

# 提取最优参数值
best_params = result.x
print(best_params)
