import math
import mne
import matplotlib.pylab as plt
import numpy as np
import torch
import torch.nn as nn
import scipy.io as io
import itertools
from sklearn.metrics import confusion_matrix
from d2l import torch as d2l
import torch
from matplotlib.ticker import MaxNLocator, MultipleLocator
import torch.nn.functional as F
from torch.nn.modules.utils import _single, _pair, _triple

def linear_combination(x, y, epsilon):
    return epsilon * x + (1 - epsilon) * y


def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss


class LabelSmoothingCrossEntropy(torch.nn.Module):
    def __init__(self, epsilon: float = 0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return linear_combination(loss / n, nll, self.epsilon)


def load_BCI_data(root, num, train=True, plot=False):
    train_all_data = io.loadmat(root + 'A0' + str(num) + 'T.mat')
    train_signals = train_all_data['data']
    train_labels = train_all_data['label']
    train_signals = (train_signals - np.mean(train_signals, axis=2, keepdims=True)) / np.std(train_signals)

    test_all_data = io.loadmat(root + 'A0' + str(num) + 'E.mat')
    test_signals = test_all_data['data']
    test_labels = test_all_data['label']
    test_signals = (test_signals - np.mean(test_signals, axis=2, keepdims=True)) / np.std(test_signals)

    return train_signals, train_labels, test_signals, test_labels


def load_data(root, sub_num, batch_size, training=True):
    train_all_data = io.loadmat(root + 'A0' + str(sub_num) + 'T.mat')
    test_all_data = io.loadmat(root + 'A0' + str(sub_num) + 'E.mat')

    train_signals = train_all_data['data'].astype(np.float32)
    train_labels = (train_all_data['label'] - 1).astype(np.float32)

    test_signals = test_all_data['data'].astype(np.float32)
    test_labels = (test_all_data['label'] - 1).astype(np.float32)
    # train_signals = np.expand_dims(train_signals, axis=1)
    # test_signals = np.expand_dims(test_signals, axis=1)
    # np.save('../visualization/train_data.npy', train_signals)
    target_mean = np.mean(train_signals)
    target_std = np.std(train_signals)

    train_signals = (train_signals - target_mean) / target_std
    test_signals = (test_signals - target_mean) / target_std

    train_signals = np.expand_dims(train_signals, axis=1)
    test_signals = np.expand_dims(test_signals, axis=1)

    shuffle_num = np.random.permutation(len(train_signals))
    train_signals = train_signals[shuffle_num, :, :, :]

    train_labels = train_labels[shuffle_num]
    train_signals = torch.from_numpy(train_signals.astype(np.float32))
    train_labels = torch.from_numpy(train_labels)
    test_signals = torch.from_numpy(test_signals.astype(np.float32))
    test_labels = torch.from_numpy(test_labels)

    train_dataset = torch.utils.data.TensorDataset(train_signals, train_labels.squeeze(1))
    test_dataset = torch.utils.data.TensorDataset(test_signals, test_labels.squeeze(1))
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                                               drop_last=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True,
                                              drop_last=False)

    return train_loader, test_loader, len(train_dataset), len(test_dataset), test_signals, test_labels

def aug_data(timg, label, batch_size):
    timg = timg.cpu().numpy()
    label = label.cpu().numpy()
    aug_data = []
    aug_label = []
    for cls4aug in range(4):
        cls_idx = np.where(label == cls4aug)
        tmp_data = timg[cls_idx]
        tmp_label = label[cls_idx]

        tmp_aug_data = np.zeros((int(batch_size / 4), 1, 22, 1000))
        for ri in range(int(batch_size / 4)):
            for rj in range(8):
                rand_idx = np.random.randint(0, tmp_data.shape[0], 8)
                tmp_aug_data[ri, :, :, rj * 125:(rj + 1) * 125] = tmp_data[rand_idx[rj], :, :, rj * 125:(rj + 1) * 125]

        aug_data.append(tmp_aug_data)
        aug_label.append(tmp_label)
    aug_data = np.concatenate(aug_data)
    aug_label = np.concatenate(aug_label)
    aug_shuffle = np.random.permutation(len(aug_data))
    aug_data = aug_data[aug_shuffle, :, :, :]
    aug_label = aug_label[aug_shuffle]

    aug_data = torch.from_numpy(aug_data).cuda()
    aug_data = aug_data.float()
    aug_label = torch.from_numpy(aug_label).cuda()
    aug_label = aug_label.long()
    return aug_data, aug_label


def plot_confusion_matrix(y_true, y_pred, sub, title="混淆矩阵 - 2a",
                          cmap=plt.cm.Blues, save_flg=True):
    y_pred = y_pred.cpu().detach().numpy()
    y_true = y_true.cpu().detach().numpy()
    classes = [str(i) for i in range(4)]
    labels = range(4)

    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize='true')
    plt.figure(figsize=(14, 12))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=40)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=20)
    plt.yticks(tick_marks, classes, fontsize=20)

    # print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], '.2f'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=30)

    plt.ylabel('真实标签', fontsize=30)
    plt.xlabel('预测标签', fontsize=30)

    if save_flg:
        plt.savefig("../pic/confusion_matrix %d" % sub + ".svg")
    plt.show()


def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(2.5, 2.5), cmap='Reds', save_fig=False,
                  channel_attention=True, sub=None):
    """Show heatmaps of matrices.

    Defined in :numref:`sec_attention-cues`"""
    d2l.use_svg_display()
    num_rows, num_cols = matrices.shape[0], matrices.shape[1]
    fig, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize,
                                 sharex=True, sharey=True, squeeze=False)
    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
            pcm = ax.imshow(d2l.numpy(matrix), cmap=cmap)
            if i == num_rows - 1:
                ax.set_xlabel(xlabel, fontsize=25)
            if j == 0:
                ax.set_ylabel(ylabel, fontsize=25)
            if titles:
                ax.set_title(titles[j], fontsize=25)
    fig.colorbar(pcm, ax=axes, shrink=0.6)
    if channel_attention:
        x_major_locator = MultipleLocator(1)
        y_major_locator = MultipleLocator(1)
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
    else:
        x_major_locator = MultipleLocator(20)
        y_major_locator = MultipleLocator(20)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
    plt.gca().xaxis.set_major_locator(x_major_locator)
    plt.gca().yaxis.set_major_locator(y_major_locator)
    if save_fig:
        plt.savefig('../pic/attention_weights%d.svg' % sub, dpi=300, bbox_inches='tight')
    plt.show()
