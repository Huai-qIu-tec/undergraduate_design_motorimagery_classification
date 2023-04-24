import numpy as np
import torch
from einops import rearrange
import matplotlib.pyplot as plt
from cam_method import GradCAM, show_cam_on_image
import mne
import sys
import scipy.io as io
sys.path.append(r'E:\_undergraduate design\source code\model')
from CNNTransformer import EEGCNNTransformer

plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']


def cam(data, sub):
    # reshape_transform  b 20 40 -> b 40 1 20
    def reshape_transform(tensor):
        result = rearrange(tensor, 'b (h w) e -> b e (h) (w)', h=1)
        return result

    device = torch.device("cpu")
    model = EEGCNNTransformer(channels=20)

    checkpoint = torch.load('../model_state_dict/conformer/conformer_sub' + str(sub) + '.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    target_layers = [model.transformer.blks[0].norm1]  # set the target layer
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False, reshape_transform=reshape_transform)

    # TODO: Class Activation Topography (proposed in the paper)

    biosemi_montage = mne.channels.make_standard_montage('biosemi64')
    # for bci competition iv 2a
    index = [37, 9, 10, 46, 45, 44, 13, 12, 11, 47, 48, 49, 50, 17, 18, 31, 55, 54, 19, 30, 56, 29]
    biosemi_montage.ch_names = [biosemi_montage.ch_names[i] for i in index]
    biosemi_montage.dig = [biosemi_montage.dig[i + 3] for i in index]
    info = mne.create_info(ch_names=biosemi_montage.ch_names, sfreq=250., ch_types='eeg')

    all_cam = []
    # this loop is used to obtain the cam of each trial/sample
    for i in range(72):
        test = torch.as_tensor(data[i:i + 1, :, :, :], dtype=torch.float32)
        test = torch.autograd.Variable(test, requires_grad=True)
        grayscale_cam = cam(input_tensor=test, target_category=target_category)
        grayscale_cam = grayscale_cam[0, :]
        all_cam.append(grayscale_cam)

    # the mean of all data
    test_all_data = np.squeeze(np.mean(data, axis=0))
    # test_all_data = (test_all_data - np.mean(test_all_data)) / np.std(test_all_data)
    mean_all_test = np.mean(test_all_data, axis=1)

    # the mean of all cam
    test_all_cam = np.mean(all_cam, axis=0)
    test_all_cam = (test_all_cam - np.mean(test_all_cam)) / np.std(test_all_cam)
    mean_all_cam = np.mean(test_all_cam, axis=1)

    # apply cam on the input data
    hyb_all = test_all_data * test_all_cam
    hyb_all = (hyb_all - np.mean(hyb_all)) / np.std(hyb_all)
    mean_hyb_all = np.mean(hyb_all, axis=1)

    evoked = mne.EvokedArray(test_all_data, info)
    evoked.set_montage(biosemi_montage)
    return mean_all_test, mean_hyb_all, evoked.info


mean_all_list = []
mean_cam_list = []
info_list = []
for j in range(1):
    nSub = j + 1
    train_test = True  # train True test False

    for i in range(1):


        target_category = i  # set the class (class activation mapping)
        if train_test:
            root = '../../数据集/BCICIV_2a_mat/A0' + str(nSub) + 'T.mat'
        else:
            root = '../../数据集/BCICIV_2a_mat/A0' + str(nSub) + 'E.mat'
        all_data = io.loadmat(root)
        labels = (all_data['label'] - 1).astype(np.float32).squeeze(1)
        category = [labels == target_category]
        data = all_data['data'].astype(np.float32)
        mean = np.mean(data)
        std = np.std(data)
        data = (data - mean) / std
        data = data[category]
        data = np.expand_dims(data, 1)
        print(data.shape)

        mean_all_test, mean_hyb_all, info = cam(data, nSub)
        mean_all_list.append(mean_all_test)
        mean_cam_list.append(mean_hyb_all)
        info_list.append(info)

    fig, axes = plt.subplots(2, 4, figsize=(10, 6))

    for i in range(4):
        mne.viz.plot_topomap(mean_all_list[i], info_list[i], show=False, axes=axes[0, i], res=1200)
        mne.viz.plot_topomap(mean_cam_list[i], info_list[i], show=False, axes=axes[1, i], res=1200)

    import pandas as pd

pd.DataFrame(mean_cam_list).to_excel('cam_22channels.xlsx')
fig.tight_layout()
plt.savefig('../../pic/需要用的图/sub%d四个类别的加权地形图.svg' % nSub, dpi=600)
plt.show()