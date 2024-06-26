import pymc3 as pm
import numpy as np

# 合并观测数据和自变量
# x1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # 用自己的数据替换
# x2 = np.array([0.5, 1.0, 1.5, 2.0, 2.5])  # 用自己的数据替换
# x3 = np.array([0.2, 0.3, 0.4, 0.5, 0.6])  # 用自己的数据替换
# x4 = np.array([0.1, 0.2, 0.3, 0.4, 0.5])  # 用自己的数据替换
# y1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # 用自己的数据替换
# y2 = np.array([0.5, 1.0, 1.5, 2.0, 2.5])  # 用自己的数据替换

# 创建 PyMC3 模型
with pm.Model() as model:
    # 定义参数的先验分布
    a = pm.Uniform("a", lower=-200, upper=200)
    b = pm.Uniform("b", lower=-2 * np.pi, upper=2 * np.pi)
    # c = pm.Uniform("c", lower=-300, upper=300)

    # 定义模型
    # y1_pred = a * x1 * pm.math.cos(b) + x2 * c + x3
    # y2_pred = -a * pm.math.sin(b) + x4 * c
    y1_pred = a * x1 * pm.math.cos(b) + x2 + x3
    y2_pred = -a * pm.math.sin(b) + x4

    # 定义似然函数
    sigma1 = pm.HalfNormal("sigma1", sigma=20)  # y1 的标准差
    sigma2 = pm.HalfNormal("sigma2", sigma=20)  # y2 的标准差
    y1_obs = pm.Normal("y1_obs", mu=y1_pred, sigma=sigma1, observed=y1)
    y2_obs = pm.Normal("y2_obs", mu=y2_pred, sigma=sigma2, observed=y2)

# 运行 MCMC 采样
with model:
    trace = pm.sample(2000, tune=1000, cores=4)

# 绘制后验分布
# pm.traceplot(trace)
# plt.show()

# 输出参数估计的后验分布
# pm.summary(trace, var_names=["a", "b", "c"])
pm.summary(trace, var_names=["a", "b"])
