import matplotlib.pyplot as plt
import numpy as np
import palettable as palettable
import pandas as pd
import scipy.stats
import seaborn as sns

plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']

no_transformer_augment = [75.69, 49.49, 86.10, 56.59, 61.28, 57.99, 80.39, 80.04, 78.81]
no_data_augment = [82.11, 49.13, 90.62, 64.07, 63.37, 59.20, 82.47, 83.86, 80.72]
no_transformer = [76.73, 54.54, 89.58, 60.41, 59.37, 57.98, 88.72, 82.65, 83.33]
transformer = [84.21, 62.85, 93.57, 68.23, 65.8, 63.89, 95.66, 90.11, 85.59]

model_acc = [[76.00, 56.50, 81.25, 61.00, 55.00, 45.25, 82.75, 81.25, 70.75],
             [80.00, 65.30, 87.10, 67.50, 55.50, 50.10, 91.70, 84.10, 87.80],
             [85.76, 61.46, 88.54, 67.01, 55.9, 52.08, 89.58, 83.33, 86.81],
             [79.00, 49.80, 90.00, 60.30, 70.80, 50.80, 79.70, 81.80, 77.40],
             [76.39, 55.21, 89.24, 74.65, 56.94, 54.17, 92.71, 77.08, 76.39],
             [77.39, 60.14, 82.92, 72.28, 75.83, 68.98, 76.03, 76.85, 84.66],
             [87.50, 65.28, 90.28, 66.67, 62.50, 45.49, 89.58, 83.33, 79.51],
             [85.42, 62.85, 93.57, 68.23, 65.8, 63.89, 95.66, 90.11, 85.59]]

std = [[2.61, 2.75, 2.38, 2.45, 1.72, 2.1, 0.53, 1.39, 3.76],
       [2.58, 2.62, 2.47, 3.45, 0.73, 1.74, 1.97, 2.59, 2.95],
       [1.91, 3.66, 3.32, 2.21, 3.13, 3.88, 3.92, 2.68, 4.81],
       [2.32, 4.5, 3.91, 2.05, 2.57, 1.61, 2.71, 3.63, 4.09], ]

print('ttest rel: \t', scipy.stats.ttest_rel(transformer, no_transformer))
print('Wilcoxon SignedRank Test : \t ', scipy.stats.wilcoxon(transformer, no_transformer))

print('ttest rel: \n', scipy.stats.ttest_rel(transformer, no_data_augment))
print('Wilcoxon SignedRank Test : \t ', scipy.stats.wilcoxon(transformer, no_data_augment))

for i in model_acc[0:-1]:

    print('Wilcoxon SignedRank Test : \t ', scipy.stats.wilcoxon(i, model_acc[-1]))

no_transformer_fold = pd.DataFrame([[79.31, 76.52, 72.17, 76.52, 79.13],
                                    [50.86, 53.91, 53.04, 58.26, 56.62],
                                    [92.24, 85.22, 88.70, 91.30, 90.43],
                                    [62.07, 53.91, 60.00, 63.48, 62.61],
                                    [60.34, 59.13, 60.00, 59.13, 58.26],
                                    [61.21, 58.26, 56.52, 57.39, 56.52],
                                    [87.93, 86.96, 89.57, 92.17, 86.96],
                                    [79.31, 84.35, 80.00, 86.09, 83.48],
                                    [85.34, 81.74, 85.22, 78.26, 86.09]]).T

transformer_fold = pd.DataFrame([[87.91, 86.12, 84.34, 81.77, 80.93],
                                 [62.07, 66.09, 66.09, 60.00, 60.00],
                                 [95.69, 90.43, 93.91, 96.52, 91.30],
                                 [68.97, 68.70, 69.57, 70.43, 63.48],
                                 [66.38, 62.61, 67.83, 66.09, 66.09],
                                 [63.79, 62.61, 65.22, 60.87, 66.96],
                                 [94.83, 95.65, 95.65, 95.65, 96.52],
                                 [88.79, 91.30, 88.70, 92.17, 89.57],
                                 [87.07, 84.35, 90.43, 79.13, 86.96]]).T

parameters_data = pd.read_excel(r'E:\_undergraduate design\实验数据.xlsx', sheet_name='参数敏感度', header=0)
conv_kernel_data = parameters_data.iloc[:5, 1:8]
conv_kernel_std = np.std(conv_kernel_data)
conv_kernel_mean = np.mean(conv_kernel_data)
plt.plot([20, 40, 60, 80, 100, 150, 200], conv_kernel_mean, marker='x')
plt.fill_between([20, 40, 60, 80, 100, 150, 200], conv_kernel_mean - conv_kernel_std,
                 conv_kernel_mean + conv_kernel_std, alpha=0.2)
plt.xlim([20, 200])
plt.ylim([75, 90])
plt.show()

# legend = ["overall", "no transformer", "no augment", "none"]
# labels = ['S' + str(i + 1) for i in range(9)]
# data = [transformer, no_transformer, no_data_augment, no_transformer_augment]
# colors = ['#C82423', '#F8AC8C', '#9AC9DB', '#2978B5']
#
# plt.figure(figsize=(12, 8))
# x = np.arange(len(labels))  # x轴刻度标签位置
# width = 0.2  # 柱子的宽度
# error_attri = {"elinewidth": 2, "ecolor": "black", "capsize": 2}
# plt.grid(axis='y', ls='--', zorder=0)
# plt.bar(x - 1.5 * width, transformer, width, label='overall', color=colors[0], zorder=10, yerr=std[0],error_kw=error_attri)
# plt.bar(x - 0.5 * width, no_transformer, width, label='no transformer', color=colors[1], zorder=10, yerr=std[1],error_kw=error_attri)
# plt.bar(x + 0.5 * width, no_data_augment, width, label='no augment', color=colors[2], zorder=10, yerr=std[2],error_kw=error_attri)
# plt.bar(x + 1.5 * width, no_transformer_augment, width, label='none', color=colors[3], zorder=10, yerr=std[3],error_kw=error_attri)
# plt.ylim([0, 125])
# plt.ylabel('准确率', fontsize=15)
# plt.xlabel('被试编号', fontsize=15)
#
# # x轴刻度标签位置不进行计算
# plt.xticks(x, labels=labels, fontsize=15)
# plt.yticks(fontsize=15)
# plt.legend(loc='upper right', fontsize=15)
# plt.savefig('../../pic/消融实验.svg', dpi=500)
# plt.show()
