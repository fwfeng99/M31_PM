Useful_function：有用的天文函数及示例

M31_rotation：旋转曲线数据，相对于星系的中心点

M31_simulation：生成本地坐标系以及天球坐标系的恒星散点，计算出理想星表（上帝视角）

M31_star_magnitude：采样已有星表数据，生成M31模拟恒星星等

M31_error：生成误差，包括星等，颜色，CTE,畸变等

M31_PM：向已经生成的模拟M31恒星的理想天球坐标中，添加误差，解自行

M31_PM_diff：原理同M31_PM，但是对照性地采用了中间历元的方法，并且对程序的计算速度进行了改进。其中目前的关键且校正的程序是
            3_1_Gaia_negative_model.py; 3_2_M31_Mag_error_PM_MCI_disk.py

M31_PM_：修正所加的系统、随机误差，计算自行值对比（策略的探究）

M31_read_simulated_image: 读取仿真图像数据，如何分析

