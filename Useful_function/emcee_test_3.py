import numpy as np
import emcee
import matplotlib.pyplot as plt


# 定义似然函数
def ln_likelihood(params, x1, x2, x3, x4, y1, y2, sigma1, sigma2):
    a, b, c = params

    # 添加对参数c的限制
    if not (250 < c < 300):
        return -np.inf  # 参数c超出范围时返回负无穷

    model_y1 = x1 * a * np.cos(b) + x2 * c + x3
    model_y2 = -a * np.sin(b) + x4 * c

    ln_likelihood_y1 = -0.5 * np.sum((y1 - model_y1) ** 2 / sigma1 ** 2)
    ln_likelihood_y2 = -0.5 * np.sum((y2 - model_y2) ** 2 / sigma2 ** 2)
    return ln_likelihood_y1 + ln_likelihood_y2


# 定义先验分布（示例中使用均匀分布）
def ln_prior(params):
    a, b, c = params
    if 0 < a < 200 and 0 < b < 2 * np.pi and 100 < c < 300:
        return 0.0  # 先验概率为常数
    return -np.inf  # 参数超出先验范围时的先验概率为零


# 定义总的对数概率函数
def ln_probability(params, x1, x2, x3, x4, y1, y2, sigma1, sigma2):
    lp = ln_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + ln_likelihood(params, x1, x2, x3, x4, y1, y2, sigma1, sigma2)


# 设置 MCMC 链的参数
ndim = 3  # 参数的数量
nwalkers = 200  # 随机行走者的数量
nsteps = 3000  # 每个行走者的步数

# 初始化行走者的起始位置
initial_params = np.array([100, np.pi / 2, 240])  # 初始参数估计
initial_pos = initial_params + 1e-3 * np.random.randn(nwalkers, ndim)

# 创建 MCMC 采样器
sampler = emcee.EnsembleSampler(nwalkers, ndim, ln_probability, args=(x1, x2, x3, x4, y1, y2, sigma1, sigma2))

# 运行 MCMC 采样
sampler.run_mcmc(initial_pos, nsteps)

# 获取采样结果
samples = sampler.chain  # 采样轨迹
samples = samples[:, :, :].reshape((-1, ndim))  # 展平

# 绘制参数的概率分布
# fig, axes = plt.subplots(ndim, figsize=(8, 6), sharex=True)
# labels = ["a", "b", "c"]
# for i in range(ndim):
#     ax = axes[i]
#     ax.hist(samples[:, i], bins=50, density=True, color="b", alpha=0.5)
#     ax.set_xlabel(labels[i])
#     ax.set_ylabel("Probability")
#     ax.axvline(np.percentile(samples[:, i], 16), color="r", linestyle="--")
#     ax.axvline(np.percentile(samples[:, i], 84), color="r", linestyle="--")
#     ax.axvline(np.percentile(samples[:, i], 50), color="r")
#
# plt.tight_layout()
# plt.show()

# 输出参数估计的中位数和不确定性范围
a_median, b_median, c_median = np.percentile(samples, 50, axis=0)
a_err = (np.percentile(samples, 84, axis=0) - np.percentile(samples, 16, axis=0)) / 2

print(f"a = {a_median:.3f} +/- {a_err[0]:.3f}")
print(f"b = {b_median:.3f} +/- {a_err[1]:.3f}")
print(f"c = {c_median:.3f} +/- {a_err[2]:.3f}")
