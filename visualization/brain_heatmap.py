import numpy as np
import torch
import matplotlib.pyplot as plt
import mne
import sys
import scipy.io as io
from matplotlib.ticker import MultipleLocator
import seaborn as sns
import pandas as pd

sys.path.append(r'E:\_undergraduate design\source code\model')
sys.path.append(r'E:\_undergraduate design\source code\tools')
from CNNTransformer import EEGCNNTransformer
from utils import load_data, show_heatmaps


def predict(net, predict_data, true_label, num_layers, num_heads, num_queries, sub):
    net.cpu()
    if predict_data.dim() == 4:
        predict_data = predict_data.cpu()
        output = net(predict_data)
        predict_label = torch.max(output, dim=1)[1]
        attention_weigths = torch.cat(
            [i.reshape((72, -1, num_queries, num_queries)).mean(dim=0) for i in net.transformer.attention_weights],
            dim=0).reshape((num_layers, num_heads, -1, num_queries))
        # show_heatmaps(attention_weigths.cpu(), xlabel='Key 时间段', ylabel='Query 时间段',
        #                titles=['头 %d' % i for i in range(1, 11)], figsize=(12, 9), channel_attention=False, sub=sub)
        # show_heatmaps(net.channel_attention.attention_weights.mean(dim=0, keepdim=True).cpu(), xlabel='Key 通道', ylabel='Query 通道',
        #               titles=['头 %d' % i for i in range(1, 11)], figsize=(12, 9), save_fig=True, channel_attention=True, sub=sub)
        return true_label, predict_label, attention_weigths


plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']

root = '../../数据集/BCICIV_2a_mat/'
batch_size = 288
net = EEGCNNTransformer(channels=20)
diagonal_weights = []
target_category = 3
for i in range(9):
    print('sub %d' % (i + 1))
    nSub = i + 1
    checkpoint = torch.load('../model_state_dict/conformer/conformer_sub' + str(nSub) + '.pth')
    net.load_state_dict(checkpoint['model_state_dict'])
    _, _, _, _, test_data, test_labels = load_data(root, sub_num=nSub, batch_size=batch_size, training=True)
    test_data = torch.unsqueeze(test_data[test_labels == target_category], 1)
    test_labels = test_labels[test_labels == target_category]
    # CNN 6 10 61 STT 3 3 190
    Y_true, Y_predict, attention_weigths = predict(net, test_data, test_labels, 6, 10, 20, nSub)
    weights = torch.mean(attention_weigths, dim=[0, 1])
    diagonal_weights.append([weights[i, i].detach().numpy() for i in range(20)])

diagonal_weights = np.array(diagonal_weights)
# plt.plot(diagonal_weights)
# x_major_locator = MultipleLocator(1)
# ax = plt.gca()
# ax.xaxis.set_major_locator(x_major_locator)
# plt.show()
diagonal_weights = pd.DataFrame(diagonal_weights,
                                index=range(1, 10),
                                columns=range(1, 21))

# sns.set_theme(style="whitegrid",font='TFlux Regular',font_scale=1)
plt.figure(figsize=(9, 4))
sns.heatmap(data=diagonal_weights, vmin=0.05, vmax=0.054, cmap="YlGn", linewidths=3)
plt.xlabel('时间段', fontsize=12)
plt.ylabel('被试者编号', fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.tight_layout()
plt.savefig('../../pic/需要用的图/类别' + str(target_category+1) + '时间段热力图.svg', dpi=600)
plt.show()
