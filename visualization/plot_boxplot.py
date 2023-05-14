import matplotlib.pyplot as plt
import numpy as np
import scipy.stats


plt.style.use('seaborn')
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']
facecolor = ["#3276B1", "#001F3F", "#800000"]


'''
    卷积核参数敏感度检验
'''


plt.figure(figsize=(12, 9))
plt.grid(True)

x = [20, 40, 60, 80, 100, 150, 200]
y = np.array([[84.48, 85.22, 84.35, 82.61, 83.48],
              [87.07, 86.06, 85.22, 78.26, 81.74],
              [87.07, 86.96, 87.83, 82.61, 82.61],
              [86.21, 86.96, 81.74, 78.26, 78.26],
              [85.34, 86.09, 82.61, 80.87, 80.87],
              [83.62, 86.96, 81.74, 78.26, 78.26],
              [82.76, 84.35, 85.22, 84.35, 80.00]])

plt.subplot(221)
plt.boxplot(y.T, patch_artist=True, showmeans=True,
            boxprops={"facecolor": facecolor[0],
                      "edgecolor": "grey",
                      "linewidth": 0.5,
                      "alpha": 0.75},
            medianprops={"color": "k", "linewidth": 0.5},
            meanprops={'marker': '+',
                       'markerfacecolor': 'k',
                       'markeredgecolor': 'k',
                       'markersize': 10})

for step, i in enumerate(y):
    if step == 2:
        continue
    print('Wilcoxon SignedRank Test : \t ', scipy.stats.wilcoxon(i, y[2]))

plt.xticks(range(1, y.shape[0] + 1), x, fontsize=12)
plt.yticks(fontsize=12)
plt.ylabel('准确率', fontsize=12)
plt.xlabel('卷积核大小', fontsize=12)
# plt.tight_layout()
# plt.savefig('../../pic/需要用的图/卷积核参数敏感度检验.svg', dpi=600)
# plt.show()

'''
    transformer头数量参数敏感度检验
'''
print('transformer头数量参数敏感度检验')

x = [1, 2, 4, 5, 8, 10, 20, 40]
y = [[84.48, 86.96, 85.22, 80.87, 83.48],
     [85.34, 87.83, 86.09, 81.74, 79.13],
     [87.07, 86.96, 87.83, 82.61, 82.61],
     [84.48, 86.96, 85.22, 82.61, 82.61],
     [84.48, 86.96, 84.35, 81.74, 83.48],
     [87.91, 86.12, 84.34, 81.77, 80.93],
     [85.34, 86.96, 86.09, 82.61, 81.74],
     [85.34, 85.22, 83.48, 83.48, 82.61]]


plt.subplot(222)
plt.boxplot(y, patch_artist=True, showmeans=True,
            boxprops={"facecolor": facecolor[1],
                      "edgecolor": "grey",
                      "linewidth": 0.5,
                      "alpha": 0.75},
            medianprops={"color": "k", "linewidth": 0.5},
            meanprops={'marker': '+',
                       'markerfacecolor': 'k',
                       'markeredgecolor': 'k',
                       'markersize': 10})

plt.xticks(range(1, len(y)+1), x, fontsize=12)
plt.yticks(fontsize=12)
plt.ylabel('准确率', fontsize=12)
plt.xlabel('Transformer模块头数量', fontsize=12)
# plt.tight_layout()
# plt.savefig('../../pic/需要用的图/transformer头数量参数敏感度检验.svg', dpi=600)
# plt.show()

for step, i in enumerate(y):
    if step == 5:
        continue
    print('Wilcoxon SignedRank Test : \t ', scipy.stats.wilcoxon(i, y[5]))

'''
    transformer重复次数参数敏感度检验
'''

print('transformer重复次数参数敏感度检验')

x = range(1, 11, 1)
y = [[87.07, 86.96, 85.22, 83.48, 80.87],
     [87.07, 85.22, 84.35, 81.74, 82.61],
     [84.48, 84.35, 85.22, 80.87, 81.74],
     [86.21, 85.22, 85.22, 83.48, 83.48],
     [84.48, 86.96, 87.83, 87.83, 81.74],
     [87.07, 86.96, 87.83, 82.61, 82.61],
     [84.48, 85.22, 86.96, 83.48, 83.48],
     [85.34, 85.22, 86.09, 82.61, 82.61],
     [83.62, 86.09, 83.48, 80.87, 80.00],
     [85.34, 86.09, 86.09, 85.22, 80.00]]

plt.subplot(212)
plt.boxplot(y, patch_artist=True, showmeans=True,
            boxprops={"facecolor": facecolor[2],
                      "edgecolor": "grey",
                      "linewidth": 0.5,
                      "alpha": 0.75},
            medianprops={"color": "k", "linewidth": 0.5},
            meanprops={'marker': '+',
                       'markerfacecolor': 'k',
                       'markeredgecolor': 'k',
                       'markersize': 10})

plt.xticks(range(1, len(y)+1), x, fontsize=12)
plt.yticks(fontsize=12)
plt.ylabel('准确率', fontsize=12)
plt.xlabel('Transformer模块数量', fontsize=12)
plt.tight_layout()
plt.savefig('../../pic/需要用的图/transformer参数敏感度检验.svg', dpi=600)

plt.show()

for step, i in enumerate(y):
    if step == 5:
        continue
    print('Wilcoxon SignedRank Test : \t ', scipy.stats.wilcoxon(i, y[5]))