"""
2CET-GAN
Copyright (c) 2022-present, Xiaohang Hu.
This work is licensed under the MIT License.
"""

import matplotlib.pyplot as plot
import numpy as np

# best 65000
fid_e_z_test = [
    # 209.49640020343276,
    # 90.13784098521678,
    # 66.7783783739523,
    # 50.61916716918671,
    45.40153095988977, 42.02587080286734, 38.48608350030066, 35.74761965188783, 34.8374253895639,
    35.10858298202486, 34.886617074017195, 34.88173683988904, 35.60101703148692, 32.7709442543763,
    32.51780038612587, 33.06943208825935, 31.49965912892243, 33.54485212377668, 32.63023028511299,
    32.42273963359611, 34.39375883109907, 34.185843485073235
    # , 33.54861554010699, 33.07225787991828
]
fid_e_z_train = [
    # 180.64913697172292,
    # 49.0951014281338,
    # 26.982413603420902,
    # 17.96788716605109,
    14.389116611605246,
    10.327506284753088, 8.93614472886772, 7.97174130157074, 7.203114136746013,
    6.511467317694171, 6.206939843123858, 5.827496837438539, 5.5674733625717465, 5.375620390691888,
    5.333883791835214, 5.122374692604649, 5.145950970313898, 4.944552692628619, 4.7985552721640286,
    4.904156425768583, 4.666293165813877, 4.802750006285805
    # , 4.629637997678746, 4.745738421840989
]
fid_e_r_test = [
    # 215.73435829579162,
    # 99.37105667008622,
    # 75.37189916028981,
    # 57.40352423009552,
    53.15893741325095,
    45.70829796176702, 40.22809025812783, 37.994097652517326, 35.362948144339846,
    34.904624036037866, 33.215331155089885, 35.08809517983582, 34.544438277007444, 32.49199499975323,
    32.945748022696655, 32.230245847336064, 31.332979344708132, 32.58626940643732, 33.02401338603422,
    31.85628069027087, 34.15446737298164, 33.028986606913065
    # , 33.05722091066797, 32.93279148699739
]
fid_e_r_train = [
    # 184.71063412533866,
    # 55.59534440808823,
    # 31.641687058881587,
    # 22.589590856441866,
    19.984306097327078,
    11.95268603020407, 9.99470229681804, 8.840181393439858, 7.83041106333709,
    6.979170935048143, 6.646222179651412, 6.2150403027287275, 5.9408015308588045, 5.769438219697127,
    5.607093702432581, 5.476119844236746, 5.467026760670127, 5.2313690487217155, 5.11844873224252,
    5.099396768590824, 4.995207567026394, 5.111304281459599
    # , 4.896371378238067, 5.026092020641143
]

# fid_e_z
y_test = fid_e_z_test
y_train = fid_e_z_train

# y_test = fid_e_r_test
# y_train = fid_e_r_train

n = 12
n = len(fid_e_r_train)
y_test = y_test[:n]
y_train = y_train[:n]

print(len(y_test))
x0 = np.array([
    # 1000
    # , 2000
    #    , 3000
    # , 4000
], np.float64)
n1 = len(y_test) - len(x0)
x1 = np.linspace(5000, 5000 * n1, n1)
print(x0.dtype)
x = np.concatenate((x0, x1))
print(x)

font_size = 16
# y2 = [c_y1(i) for i in x]
plot.figure(figsize=(10, 7))
# plot.plot(x, y_test, label="test set", color='#DF732C')
plot.plot(x, y_test, label="test set", color='black', linestyle='--', marker='o')
# plot.plot(x, y_train, label="train set", color='#42946B')
plot.plot(x, y_train, label="train set", color='black', linestyle='-', marker='o')
plot.legend(loc='right', frameon=False, fontsize=font_size)
plot.xticks(x, fontsize=font_size, rotation=0)
# plot.gca().set_yticklabels(plot.gca().get_yticks(), fontsize=12, rotation=0)
plot.yticks(fontsize=font_size)
plot.xlabel('Training steps', fontsize=font_size, fontweight='bold')
plot.ylabel('FID', fontsize=font_size, fontweight='bold')

# after plotting the dataset, format the labels
current_values = plot.gca().get_xticks()
# using format string '{:.0f}' here but you can choose others
xticklabels = []
for x in current_values:
    if x % 10000 != 0:
        l = "{:.0f}K".format(x / 1000)
    else:
        l = ""
    xticklabels.append(l)
plot.gca().set_xticklabels(xticklabels)
# plot.axis('equal')

ax = plot.gca()
# ax为两条坐标轴的实例
# ax.xaxis.set_major_locator(MultipleLocator(1))
# 把x轴的主刻度设置为1的倍数
# ax.yaxis.set_major_locator(MultipleLocator(1))
# plot.xlim(-5, 10)
# plot.ylim(-1, 1)

# 去掉边框
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')

# fig = plot.figure()
# fig.tight_layout()
plot.subplots_adjust(left=0.08, right=0.9, top=0.95, bottom=0.15)

# 移位置 设为原点相交
# ax.xaxis.set_ticks_position('bottom')
# ax.spines['bottom'].set_position(('dataset', 0))
# ax.yaxis.set_ticks_position('left')
# ax.spines['left'].set_position(('dataset', 0))
plot.savefig('FID_line.png')
plot.show()
