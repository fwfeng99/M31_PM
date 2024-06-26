import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# plt.style.use('seaborn-whitegrid')

cat_err_f = 1
obs_times = [1, 5, 10, 20, 50]
part_indexs = [1, 2, 3, 4]
part_indexs = [4]

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))

ra_pm_un = []
dec_pm_un = []
for obs_time in obs_times:
    in_path_obs = 'M31_PM/PM_calculate/Data/PM_result_submit_final'
    u_ra_obs = np.array([])
    u_dec_obs = np.array([])
    for part_index in part_indexs:
        filename_obs = f'res_1_{obs_time}_{part_index}_3_val.csv'
        data_obs = pd.read_csv(os.path.join(in_path_obs, filename_obs))
        u_ra_obs = np.hstack((u_ra_obs, data_obs['ra_pm_obs'] * 0.734))
        u_dec_obs = np.hstack((u_dec_obs, data_obs['dec_pm_obs']))

    ra_pm_un.append(np.std(u_ra_obs))
    dec_pm_un.append(np.std(u_dec_obs))

ax.scatter(obs_times, ra_pm_un, s=10, c='black')
ax.scatter(obs_times, dec_pm_un, s=10, c='black')
obj_ra = ax.plot(obs_times, ra_pm_un, c='blue', label=r'$\Delta\mu_{\alpha*}$')
obj_dec = ax.plot(obs_times, dec_pm_un, c='red', label=r'$\Delta\mu_\delta$')

ax.set_xlabel('Observational times', fontweight='bold', fontsize=14)
ax.set_ylabel(r'$\sigma\mu_\alpha*$ or $\sigma\mu_\delta$ [$\mu$as/yr]', fontweight='bold', fontsize=14)
ax.tick_params(axis='x', labelsize=12, width=2)
ax.tick_params(axis='y', labelsize=12, width=2)

# ax.text(0.25, 0.9, '(' + chr(94 + acc_index) + ')', transform=ax.transAxes, fontsize=16, fontweight='bold')
ax.legend(loc='upper right', fontsize=14)

plt.gcf().set_size_inches(6, 6)  # Set figure size (width, height) in inches
plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)  # Adjust margins
path_fig = 'M31_PM/_plot/figure'
filename_fig = 'PM_obs_uncertainty.jpg'
plt.show()
# plt.savefig(os.path.join(path_fig, filename_fig), dpi=300)
