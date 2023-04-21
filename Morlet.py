import math
import mne
import matplotlib.pylab as plt
import pywt
import numpy as np
import scipy.io as io



root = '../数据集/BCICIV_2a_mat/'

sub_num = 1

train_all_data = io.loadmat(root + 'A0' + str(sub_num) + 'T.mat')
test_all_data = io.loadmat(root + 'A0' + str(sub_num) + 'E.mat')

train_signals = train_all_data['data'].astype(np.float32)
train_labels = (train_all_data['label'] - 1).astype(np.float32).squeeze(1)

test_signals = test_all_data['data'].astype(np.float32)
test_labels = (test_all_data['label'] - 1).astype(np.float32).squeeze(1)

target_mean = np.mean(train_signals)
target_std = np.std(train_signals)

train_signals = (train_signals - target_mean) / target_std
test_signals = (test_signals - target_mean) / target_std

print(train_signals.shape, train_labels.shape, test_signals.shape, test_labels.shape)

coef, freqs = pywt.cwt(train_signals[0, :, :], scales=np.arange(1, 23), wavelet='gaus1')

print(coef.shape, freqs.shape)
print(coef)