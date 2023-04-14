import os
from sklearn.preprocessing import OneHotEncoder
import pylab as plt
import warnings
from utils import load_data
from CNNTransformer import EEGCNNTransformer
import numpy as np
import torch
warnings.filterwarnings('ignore')
from torch.nn.functional import softmax
from sklearn.metrics import roc_curve, auc
from scipy import interp
from itertools import cycle

# 画图部分
plt.figure()
fpr_mean = []
tpr_mean = []
auc_mean = []


for i in range(9):
    scores = []
    test_labels = []
    nSub = i + 1
    num_class = 4
    batch_size = 72
    root = '../../数据集/BCICIV_2a_mat/'
    train_loader, test_loader, train_num, test_num, _, _ = load_data(root, sub_num=nSub, batch_size=batch_size, training=False)
    model_state_dict = '../model_state_dict/CNNTransformer_aug_sub%d.pth' % nSub
    net = EEGCNNTransformer(channels=61).cuda()
    net.eval()
    if os.path.exists(model_state_dict):
        state_dict = torch.load(model_state_dict)
        net.load_state_dict(state_dict)

    with torch.no_grad():
        for signals, labels in test_loader:
            outputs = net(signals.cuda())
            scores.extend(torch.nn.functional.softmax(outputs).detach().cpu().numpy())
            test_labels.extend(labels.numpy())

    scores = np.array(scores)

    # scores = net(test_signals).detach().numpy()
    test_labels = OneHotEncoder().fit_transform(np.array(test_labels).reshape(-1, 1)).toarray()
    print('socre shape', scores.shape)
    print('label shape', test_labels.shape)

    fpr_dict = dict()
    tpr_dict = dict()
    roc_auc_dict = dict()
    for i in range(num_class):
        fpr_dict[i], tpr_dict[i], _ = roc_curve(test_labels[:, i], scores[:, i])
        roc_auc_dict[i] = auc(fpr_dict[i], tpr_dict[i])
    # micro
    fpr_dict["micro"], tpr_dict["micro"], _ = roc_curve(test_labels.ravel(), scores.ravel())
    roc_auc_dict["micro"] = auc(fpr_dict["micro"], tpr_dict["micro"])

    # macro
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in range(num_class)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_class):
        mean_tpr += interp(all_fpr, fpr_dict[i], tpr_dict[i])
    # Finally average it and compute AUC
    mean_tpr /= num_class
    fpr_dict["macro"] = all_fpr
    tpr_dict["macro"] = mean_tpr
    roc_auc_dict["macro"] = auc(fpr_dict["macro"], tpr_dict["macro"])
    fpr_mean.append(fpr_dict["micro"])
    tpr_mean.append(tpr_dict["micro"])
    auc_mean.append(roc_auc_dict["micro"])
    # 绘制所有类别平均的roc曲线

    lw = 2
    plt.plot(fpr_dict["micro"], tpr_dict["micro"], label='Sub0{0} (area = {1:0.2f})'.format(nSub, roc_auc_dict["micro"]), linewidth=2, alpha=0.75)
    plt.fill_between(fpr_dict["micro"], tpr_dict["micro"], alpha=0.01)
#
# plt.plot(fpr_dict["macro"], tpr_dict["macro"],
#          label='macro-average ROC curve (area = {0:0.2f})'
#                ''.format(roc_auc_dict["macro"]),
#          color='navy', linestyle=':', linewidth=4)
#
# colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
# for i, color in zip(range(num_class), colors):
#     plt.plot(fpr_dict[i], tpr_dict[i], color=color, lw=lw,
#              label='ROC curve of class {0} (area = {1:0.2f})'
#                    ''.format(i, roc_auc_dict[i]))


plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.savefig('AllSub_roc.png', dpi=600)
plt.show()
