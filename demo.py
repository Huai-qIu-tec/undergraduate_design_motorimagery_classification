import sys
import gradio as gr
import random
import mne
import numpy as np
import torch
from einops import rearrange
from matplotlib import pyplot as plt
from scipy import io
import seaborn as sns
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
    Colors = ['darkred', 'navy', 'darkgreen', 'slategray', 'purple']

    # 设置画布大小
    fig = plt.figure(figsize=(25, 15))

    # 去掉横坐标和纵坐标
    plt.axis('off')

    # 绘制22个通道的波形图，每个通道颜色不同

    for i in range(22):
        color = random.choice(Colors)
        plt.plot(samples[i] - i * 15, color=color, linewidth=1)
        plt.subplots_adjust(hspace=0.5)

    plt.tight_layout()

    return fig


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

    return fig


def show_time_pic(sub, number):
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
    model(data)

    attention_weigths = torch.cat(
        [i.reshape((1, -1, 20, 20)).mean(dim=0) for i in model.transformer.attention_weights],
        dim=0).reshape((6, 10, -1, 20))

    weights = torch.mean(attention_weigths, dim=[0, 1])
    diagonal_weights = [weights[i, i].detach().numpy() for i in range(20)]

    min_weight, max_weight = min(diagonal_weights), max(diagonal_weights)
    diagonal_weights = np.expand_dims(np.array(diagonal_weights), axis=0)

    fig = plt.figure(figsize=(10, 4))

    sns.heatmap(data=diagonal_weights, vmin=min_weight, vmax=max_weight, cmap="YlGn", linewidths=3)
    plt.xlabel('time period', fontsize=12)
    plt.xticks(np.arange(1, 21, 1))
    plt.axvline(x=5, color='k', linestyle='--')
    plt.axvline(x=10, color='k', linestyle='--')
    plt.axvline(x=15, color='k', linestyle='--')
    plt.axvline(x=20, color='k', linestyle='--')
    plt.text(1.5, 0, 'cue(1s)', color='k')
    plt.text(5.5, 0, 'motor imaging(2s)', color='k')
    plt.text(10.5, 0, 'motor imaging(3s)', color='k')
    plt.text(15.5, 0, 'motor imaging(4s)', color='k')
    return fig



with gr.Blocks() as classify_demo:
    gr.Markdown("<center><span style='font-size:30px;'>基于Transformer的运动想象脑电信号分类器</span></<center>")
    with gr.Row():
        with gr.Box():
            with gr.Column():
                sub_input = gr.Slider(1, 9, label="被试者编号(1-9)", step=1)
                trail_input = gr.Slider(1, 288, label="试验编号(1-288)", step=1)
                block_number = gr.Slider(1, 6, label="地形图展示层数(1-6)", step=1)
                outputs = gr.Plot()
                show_button = gr.Button("显示脑电信号")

                time_outputs = gr.Plot()
                show_time_outputs = gr.Button("显示时间激活图")
                # outputs.style(height=500)



        show_button.click(fn=show_pic, inputs=[sub_input, trail_input], outputs=[outputs])

        with gr.Box():
            with gr.Column():
                label = gr.Label(num_top_classes=4)
                true_label = gr.Textbox(label="真实标签")
                predict_button = gr.Button("预测")

                cam_outputs = gr.Plot()

                show_cam_button = gr.Button("显示地形图")

        predict_button.click(fn=predict, inputs=[sub_input, trail_input], outputs=[label, true_label])
        show_cam_button.click(fn=show_cam_pic, inputs=[sub_input, trail_input, block_number], outputs=cam_outputs)
        show_time_outputs.click(fn=show_time_pic, inputs=[sub_input, trail_input], outputs=time_outputs)

classify_demo.launch(share=True, server_port=8080)
