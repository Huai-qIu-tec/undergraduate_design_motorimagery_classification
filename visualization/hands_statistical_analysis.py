import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats


plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']

weights = pd.read_excel('cam_22channels.xlsx', header=0)
print(weights.shape)
j = 1
for i in range(0, 36, 4):
    print('sub', j)
    j += 1
    left_hands_left_brain_weights = weights.iloc[i, [1, 2, 6, 7, 8, 13, 14, 18]]
    left_hands_right_brain_weights = weights.iloc[i, [4, 5, 10, 11, 12, 16, 17, 20]]

    right_hands_left_brain_weights = weights.iloc[i + 1, [1, 2, 6, 7, 8, 13, 14, 18]]
    right_hands_right_brain_weights = weights.iloc[i + 1, [4, 5, 10, 11, 12, 16, 17, 20]]

    print(left_hands_left_brain_weights.sum(), left_hands_right_brain_weights.sum(), scipy.stats.wilcoxon(left_hands_left_brain_weights, left_hands_right_brain_weights))
    print(right_hands_left_brain_weights.sum(),right_hands_right_brain_weights.sum(), scipy.stats.wilcoxon(right_hands_left_brain_weights, right_hands_right_brain_weights))
