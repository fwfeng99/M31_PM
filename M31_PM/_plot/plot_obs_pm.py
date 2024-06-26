import numpy as np
import pandas as pd
import matplotlib
import os
import matplotlib.pyplot as plt
print(matplotlib.get_backend())
plt.style.use('seaborn-whitegrid')


in_path = 'D:/Code/Python/M31_Kinematics/M31_PM/PM_calculate/Data/PM_result_submit'
filename = 'res_1_10_4_1_val.csv'
data = pd.read_csv(os.path.join(in_path, filename))

#

u_ra_ideal = data['ra_pm_ideal']
# u_ra_ideal_err = data['ra_pm_ideal_uncer']
u_ra_obs = data['ra_pm_obs']
# u_ra_obs_err = data['ra_pm_obs_uncer']

u_dec_ideal = data['dec_pm_ideal']
# u_dec_ideal_err = data['dec_pm_ideal_uncer']
u_dec_obs = data['dec_pm_obs']
# u_dec_obs_err = data['dec_pm_obs_uncer']

print(np.mean(u_ra_ideal) - np.mean(u_ra_obs))
print(np.mean(u_dec_ideal) - np.mean(u_dec_obs))
print('ra_err', np.std(u_ra_obs))
print('dec_err', np.std(u_dec_obs))


x = np.linspace(1, len(u_ra_ideal), len(u_ra_ideal))

fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(1, 2, 1)

plt.plot(x, u_ra_ideal, '.', color='black', label='ideal ra PM')
plt.plot(x, u_ra_obs, '.', color='red', label='observed ra PM')

# plt.errorbar(x, u_ra_ideal, yerr=u_ra_ideal_err, fmt='o', color='black', ecolor='lightgray', elinewidth=3, capsize=0, label='ideal ra PM')
# plt.errorbar(x, u_ra_obs, yerr=u_ra_obs_err, fmt='o', color='red', ecolor='lightgray', elinewidth=3, capsize=0, label='observed ra PM')

plt.xlabel('number of running', fontweight='bold')
# plt.xticks(x)
plt.ylabel(r'PM $\mu$as / year', fontweight='bold')
plt.title('ideal and observed ra PM', fontweight='bold')
plt.legend(loc='upper right')

ax = fig.add_subplot(1, 2, 2)
# plt.errorbar(x, u_dec_ideal, yerr=u_dec_ideal_err, fmt='o', color='black', ecolor='lightgray', elinewidth=3, capsize=0, label='ideal dec PM')
# plt.errorbar(x, u_dec_obs, yerr=u_dec_obs_err, fmt='o', color='red', ecolor='lightgray', elinewidth=3, capsize=0, label='observed dec PM')

plt.plot(x, u_dec_ideal, '.', color='black', label='ideal ra PM')
plt.plot(x, u_dec_obs, '.', color='red', label='observed ra PM')


plt.xlabel('number of running', fontweight='bold')
# plt.xticks(x)
plt.ylabel(r'PM $\mu$as / year', fontweight='bold')
plt.title('ideal and observed dec PM', fontweight='bold')
plt.legend(loc='upper right')

plt.show()

# fig = plt.figure(figsize=(5, 5))
# plt.hist(u_dec_obs, bins=5)
# print(np.mean(u_ra_ideal) - np.mean(u_ra_obs))
# print(np.mean(u_dec_ideal) - np.mean(u_dec_obs))
# print('ra_err', np.std(u_ra_obs))
# print('dec_err', np.std(u_dec_obs))

# np.cos(42.75/180*np.pi)

in_path = 'D:/repos/PycharmProjects/M31_kinematics_validate/M31_PM/plot/figure/'
# plt.savefig(in_path + 'figure.png', dpi=300)
