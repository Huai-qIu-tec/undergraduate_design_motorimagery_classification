import matplotlib
import matplotlib.pyplot as plt
from sklearn import manifold
from einops import reduce
import scipy.io as io
import torch
import numpy as np
import sys
import warnings
warnings.filterwarnings('ignore')
sys.path.append(r'E:\_undergraduate design\source code\model')
from CNNTransformer import EEGCNNTransformer
from CNNTransformer_notransformer import EEGCNN_notransformer
from utils import load_data

config = {
    "font.family":'serif',
    "font.size": 15,
    "mathtext.fontset":'stix',
    "font.serif": ['SimSun'],
}
plt.rcParams.update(config)


def plt_tsne(data, label, per):
    data = data.cpu().detach().numpy()
    # data = reduce(data, 'b n e -> b e', reduction='mean')
    label = label.cpu().detach().numpy().astype(np.int32)
    colors = ['#c72e29', '#098154', '#fb832d', '#F596AA']
    tsne = manifold.TSNE(n_components=2, perplexity=per, init='pca', random_state=166)
    X_tsne = tsne.fit_transform(data)

    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)
    for i in range(X_norm.shape[0]):
        plt.scatter(X_norm[i, 0], X_norm[i, 1], color=colors[label[i]])
        plt.xticks([])
        plt.yticks([])


root = '../../数据集/BCICIV_2a_mat/'
nSub = 1
batch_size = 288
per = 35.0
train_loader, test_loader, train_num, test_num, _, _ = load_data(root, sub_num=nSub, batch_size=batch_size, training=True)
net = EEGCNNTransformer(channels=20)
state_dict = torch.load('../model_state_dict/conformer/conformer_40x300x10_samemode_sub1.pth')
net.load_state_dict(state_dict)
net.eval()
plt.figure(figsize=(12, 8))

# 带有transformer的train集
for step, data_loader in enumerate(train_loader):
    data, labels = data_loader
    out = net(data)
plt.subplot(2, 2, 1)
plt.ylabel('训练集', fontsize=15)
plt.xlabel('$\mathrm{(a)}$', fontsize=15)
plt_tsne(out, labels, per)
plt.title('具有$\mathrm{Transformer}$模块', fontsize=15)

# 带有transformer的test集
for step, data_loader in enumerate(test_loader):
    data, labels = data_loader
    out = net(data)

plt.subplot(2, 2, 3)
plt.ylabel('测试集', fontsize=15)
plt.xlabel('$\mathrm{(c)}$', fontsize=15)
plt_tsne(out, labels, per)

# 不带有transformer的train集
checkpoint = torch.load('../model_state_dict/conformer/conformer_checkpoint_notransformer_sub1.pth')
net = EEGCNN_notransformer(20)
net.load_state_dict(checkpoint['model_state_dict'])
net.eval()
for step, data_loader in enumerate(train_loader):
    data, labels = data_loader
    out = net(data)
plt.subplot(2, 2, 2)
plt.xlabel('$\mathrm{(b)}$', fontsize=15)
plt_tsne(out, labels, per)
plt.title('不具有$\mathrm{Transformer}$模块', fontsize=15)

# 不带有transformer的test集
for step, data_loader in enumerate(test_loader):
    data, labels = data_loader
    out = net(data)
plt.subplot(2, 2, 4)
plt.xlabel('$\mathrm{(d)}$', fontsize=15)
plt_tsne(out, labels, per)
plt.tight_layout()
plt.savefig('../../pic/tsne.svg', dpi=600)
plt.show()
