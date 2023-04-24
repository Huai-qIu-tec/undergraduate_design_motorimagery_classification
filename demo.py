import sys

import PIL.Image
import gradio as gr
import random
import cv2
import mne
import numpy as np
import torch
from PIL.Image import Image
from einops import rearrange
from matplotlib import pyplot as plt
from scipy import io
import io as ios

from model.CNNTransformer import EEGCNNTransformer
from tools.utils import load_data

sys.path.append(r'E:\_undergraduate design\source code\visualization')
from cam_method import GradCAM


def predict(sub, number):
    root = '../数据集/BCICIV_2a_mat/'
    model = EEGCNNTransformer(channels=20)
    checkpoint_dir = './model_state_dict/conformer/conformer_sub%d.pth' % int(sub)
    class_dict = {0: '左手', 1: '右手', 2: '双脚', 3: '舌头'}

    # 导入模型训练参数
    checkpoint = torch.load(checkpoint_dir, map_location=torch.device('cpu'))
    model_state_dict = checkpoint['model_state_dict']
    model.load_state_dict(model_state_dict)

    _, _, _, _, test_data, test_labels = load_data(root, sub_num=sub, batch_size=288, training=True)

    model.eval()
    data = test_data[int(number), :, :, :].unsqueeze(0)
    label = test_labels[int(number)].cpu().numpy()

    with torch.no_grad():
        outputs = model(data)
        probability = torch.nn.functional.softmax(outputs[0], dim=0).cpu().numpy()
        confidences = {class_dict[i]: float(probability[i]) for i in range(4)}

    return confidences, class_dict[int(label)]


def show_pic(sub, trail):
    root = '../数据集/BCICIV_2a_mat/'
    test_all_data = io.loadmat(root + 'A0' + str(sub) + 'E.mat')
    test_signals = test_all_data['data'].astype(np.float32)
    samples = test_signals[int(trail), :, :]
    ch_names = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19',
                '20', '21', '22']  # 通道名称
    sfreq = 250  # 采样率
    info = mne.create_info(ch_names, sfreq, ch_types='eeg')  # 创建信号的信息
    raw = mne.io.RawArray(samples, info)
    # 绘制 EEG 信号图
    fig = raw.plot(scalings={"eeg": 1}, show_scrollbars=False, show=False)

    plt.tight_layout()

    fig.savefig('demo_pic/%d_%d_eeg.png' % (int(sub), int(trail)), dpi=150)
    img = PIL.Image.open('demo_pic/%d_%d_eeg.png' % (int(sub), int(trail)))

    return img


def show_cam_pic(sub, number, cam_num=1):
    root = '../数据集/BCICIV_2a_mat/'
    model = EEGCNNTransformer(channels=20)
    checkpoint_dir = './model_state_dict/conformer/conformer_sub%d.pth' % int(sub)

    # 导入模型训练参数
    checkpoint = torch.load(checkpoint_dir, map_location=torch.device('cpu'))
    model_state_dict = checkpoint['model_state_dict']
    model.load_state_dict(model_state_dict)

    _, _, _, _, test_data, test_labels = load_data(root, sub_num=sub, batch_size=288, training=True)

    model.eval()
    data = test_data[int(number), :, :, :].unsqueeze(0)
    label = test_labels[int(number)].cpu().numpy()

    def reshape_transform(tensor):
        result = rearrange(tensor, 'b (h w) e -> b e (h) (w)', h=1)
        return result

    target_layers = [model.transformer.blks[int(cam_num) - 1].norm1]  # set the target layer
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False, reshape_transform=reshape_transform)

    biosemi_montage = mne.channels.make_standard_montage('biosemi64')
    # for bci competition iv 2a
    index = [37, 9, 10, 46, 45, 44, 13, 12, 11, 47, 48, 49, 50, 17, 18, 31, 55, 54, 19, 30, 56, 29]
    biosemi_montage.ch_names = [biosemi_montage.ch_names[i] for i in index]
    biosemi_montage.dig = [biosemi_montage.dig[i + 3] for i in index]
    info = mne.create_info(ch_names=biosemi_montage.ch_names, sfreq=250., ch_types='eeg')

    test = torch.autograd.Variable(data, requires_grad=True)
    grayscale_cam = cam(input_tensor=test, target_category=int(label.item()))
    grayscale_cam = grayscale_cam[0, :]

    # the mean of all data
    data = data.cpu().numpy()
    test_all_data = np.squeeze(np.squeeze(data))
    test_all_data = (test_all_data - np.mean(test_all_data, axis=1, keepdims=True)) / np.std(test_all_data, axis=1,
                                                                                             keepdims=True)

    # the mean of all cam
    test_all_cam = (grayscale_cam - np.mean(grayscale_cam, axis=1, keepdims=True)) / np.std(grayscale_cam, axis=1,
                                                                                            keepdims=True)

    # apply cam on the input data
    hyb_all = test_all_data * test_all_cam
    hyb_all = (hyb_all - np.mean(hyb_all, axis=1, keepdims=True)) / np.std(hyb_all, axis=1,
                                                                           keepdims=True)
    mean_hyb_all = np.mean(hyb_all, axis=1)

    fig, axes = plt.subplots(1, 1, figsize=(6, 6))
    evoked = mne.EvokedArray(hyb_all, info)
    evoked.set_montage(biosemi_montage)
    mne.viz.plot_topomap(mean_hyb_all, evoked.info, show=False, axes=axes, res=1200)
    plt.tight_layout()

    fig.savefig('demo_pic/%d_%d_cam.png' % (int(sub), int(number)), dpi=300)
    img = PIL.Image.open('demo_pic/%d_%d_cam.png' % (int(sub), int(number)))

    return img

with gr.Blocks() as classify_demo:
    gr.Markdown("<center><span style='font-size:50px;'>运动想象脑电信号分类器</span></<center>>")
    with gr.Row():
        with gr.Box():
            with gr.Column():
                sub_input = gr.Textbox(label="被试者编号")
                trail_input = gr.Textbox(label="试验编号")
                block_number = gr.Textbox(label="地形图展示层数")
                outputs = gr.Image()
                outputs.style(height=500)
            with gr.Column(scale=1):
                show_button = gr.Button("显示脑电信号")

        show_button.click(fn=show_pic, inputs=[sub_input, trail_input], outputs=[outputs])

        with gr.Box():
            with gr.Column():
                label = gr.Label(num_top_classes=4)
                true_label = gr.Textbox(label="真实标签")
                predict_button = gr.Button("预测")
                cam_outputs = gr.Image()
                cam_outputs.style(height=350)
                show_cam_button = gr.Button("显示地形图")


        predict_button.click(fn=predict, inputs=[sub_input, trail_input], outputs=[label, true_label])
        show_cam_button.click(fn=show_cam_pic, inputs=[sub_input, trail_input, block_number], outputs=cam_outputs)

classify_demo.launch(share=True, server_port=8080)
