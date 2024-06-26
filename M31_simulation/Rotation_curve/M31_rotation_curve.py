import os
import re
import pandas as pd

# read txt context save as data
path = "M31_simulation/Rotation_curve/Data/"
filename = "M31_rotation_curve.txt"
path_filename = os.path.join(path, filename)
model = "r"
enconding_type = "utf-8"

with open(path_filename, model, encoding=enconding_type) as f:
    data = f.read()  # 按行读取，"\n"，str
    f.close()

data = re.split(r"[\n]", data)  # 按照"\n"分割的字符，r'[]'

# data clean
data = data[5:-1]

# 2d list
row, col = len(data), len(data[0].split())
data_new = [[0] * col for _ in range(row)]
for line, index in zip(data, range(len(data))):
    line = line.split()
    data_new[index] = line

# 2d list to Dataframe
data_new = pd.DataFrame(data_new)

# data clean
data_param = data_new.iloc[:, [1, 4, 7, 8, 9]]
data_param.iloc[88, [2, 3, 4]] = [74.5, 244.9, 3.6]
data_param.iloc[94, [2, 3, 4]] = [78.2, 255.7, 4.5]
data_param = data_param[2:-1]

data_param.columns = ["radius", "PA", "inclination", "Vrot", "eVrot"]
# data_param.rename(columns={'1': 'radius', '4': 'PA', '7': 'inclination', '8': 'Vrot', '9': 'eVrot'}, inplace=True)

data_param["radius"] = pd.to_numeric(data_param["radius"])
data_param["PA"] = pd.to_numeric(data_param["PA"])
data_param["inclination"] = pd.to_numeric(data_param["inclination"])
data_param["Vrot"] = pd.to_numeric(data_param["Vrot"])
data_param["eVrot"] = pd.to_numeric(data_param["eVrot"])

radius = pd.to_numeric(data_param["radius"])
PA = pd.to_numeric(data_param["PA"])
inclination = pd.to_numeric(data_param["inclination"])
Vrot = pd.to_numeric(data_param["Vrot"])
eVrot = pd.to_numeric(data_param["eVrot"])

# save
# data_param.to_csv(path_or_buf=os.path.join(path, "M31_rotation_curve.csv"), index=False)
