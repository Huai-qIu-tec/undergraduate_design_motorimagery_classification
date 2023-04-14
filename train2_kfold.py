import os
import numpy as np
import torch
import torch.nn as nn
from matplotlib.ticker import MultipleLocator
from scipy import io
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import matplotlib.pyplot as plt
from model.Spatial_Temporal_Attention import SpatialTemporalAttention
from model.CNNTransformer import EEGCNNTransformer
from tools.utils import load_data, aug_data, plot_confusion_matrix, show_heatmaps
from torchsummary import summary
from model.CNNTransformer_notransformer import EEGCNN_notransformer
from sklearn.model_selection import KFold


plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def load_data_kfold(root, sub_num, batch_size, training=True):
    train_all_data = io.loadmat(root + 'A0' + str(sub_num) + 'T.mat')
    test_all_data = io.loadmat(root + 'A0' + str(sub_num) + 'E.mat')

    train_signals = train_all_data['data'].astype(np.float32)
    train_labels = (train_all_data['label'] - 1).astype(np.float32)

    test_signals = test_all_data['data'].astype(np.float32)
    test_labels = (test_all_data['label'] - 1).astype(np.float32)

    all_data = np.concatenate([train_signals, test_signals])
    all_labels = np.squeeze(np.concatenate([train_labels, test_labels]), axis=1)
    kf = KFold(n_splits=5, shuffle=True, random_state=512)
    train_dataloader = []
    test_dataloader = []
    test_all_signals, test_all_labels = torch.Tensor(), torch.Tensor()
    for train_index, test_index in kf.split(all_data):
        train_signals, train_labels = all_data[train_index], all_labels[train_index]
        test_signals, test_labels = all_data[test_index], all_labels[test_index]
        signals_mean, signals_std = np.mean(train_signals), np.std(train_signals)
        train_signals, test_signals = (train_signals - signals_mean) / signals_std, (test_signals - signals_mean) / signals_std
        train_signals, test_signals = np.expand_dims(train_signals, 1), np.expand_dims(test_signals, 1)
        shuffle_num = np.random.permutation(len(train_signals))
        train_signals = train_signals[shuffle_num, :, :, :]
        train_labels = train_labels[shuffle_num]
        train_signals = torch.from_numpy(train_signals.astype(np.float32))
        train_labels = torch.from_numpy(train_labels)
        test_signals = torch.from_numpy(test_signals.astype(np.float32))
        test_labels = torch.from_numpy(test_labels)
        train_dataset = torch.utils.data.TensorDataset(train_signals, train_labels)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                                                   drop_last=True)
        train_dataloader.append(train_loader)
        test_dataloader.append([test_signals, test_labels])
        test_all_signals = torch.cat([test_all_signals, test_signals])

        test_all_labels = torch.cat([test_all_labels, test_labels])
    return train_dataloader, len(train_dataset), test_signals.shape[0], test_dataloader, test_all_signals, test_all_labels


def train_epoch(net, loss, scheduler, optimizer, train_loader, train_num, batch_size, test_list, test_num):
    train_running_loss = 0.0
    train_num = 0
    acc = 0.0
    train_accurates = []
    test_accurates = []
    val_labels_ls = torch.Tensor().cuda()
    predict_labels_ls = torch.Tensor().cuda()

    net.train()

    for step, data in enumerate(train_loader):
        signals, labels = data
        signals, labels = signals.cuda(), labels.cuda().long()
        aug_signals, aug_labels = aug_data(signals, labels, batch_size)
        signals, labels = torch.cat((signals, aug_signals), dim=0), torch.cat((labels, aug_labels), dim=0)
        y_hat = net(signals)
        predict_label = torch.max(y_hat, dim=1)[1]
        acc += torch.eq(predict_label, labels).sum().item()
        l = loss(y_hat, labels)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        scheduler.step()
        train_running_loss += l.sum().item() * signals.shape[0]
        train_num += signals.shape[0]

    train_accurates.append(acc / train_num)

    # validate
    net.eval()
    acc = 0.0
    test_running_loss = 0.0
    with torch.no_grad():
        val_signals, val_labels = test_list
        val_signals = val_signals.cuda()
        val_labels = val_labels.long().cuda()
        outputs = net(val_signals)
        predict_labels = torch.max(outputs, dim=1)[1]
        l = loss(outputs, val_labels)

        acc += torch.eq(predict_labels, val_labels).sum().item()
        val_labels_ls = torch.cat([val_labels_ls, val_labels])
        predict_labels_ls = torch.cat([predict_labels_ls, predict_labels])
        test_running_loss += l.sum().item() * val_signals.size(0)
    test_accurates.append(acc / val_signals.shape[0])
    train_accurate = np.mean(train_accurates)
    val_accurate = np.mean(test_accurates)
    return train_running_loss / train_num, train_accurate, test_running_loss / val_signals.shape[0], val_accurate, val_labels_ls, predict_labels_ls


def train(net, loss, optimizer, scheduler, train_loader_list, test_loader_list, train_num, test_num, epochs, device,
          batch_size, log_write, sub, checkpoint=None):
    def weight_init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        # 也可以判断是否为conv2d，使用相应的初始化方式
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        # 是否为批归一化层
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    best_acc = 0.0
    Y_true = 0.0
    Y_predict = 0.0
    train_accurates = []
    valid_accurates = []
    train_loss_ls = []
    valid_loss_ls = []
    train_fold_acc = []
    test_fold_acc = []

    for step, (train_loader, test_list) in enumerate(zip(train_loader_list, test_loader_list)):
        checkpoint_dir = './model_state_dict/conformer/fold/kfold_kernel60_checkpoint_40x200x15_samemode_sub%d_fold%d.pth' % (sub, step+1)
        if os.path.exists(checkpoint_dir):
            checkpoint = torch.load(checkpoint_dir)
            model_state_dict = checkpoint['model_state_dict']
            net.load_state_dict(model_state_dict)
        else:
            net.apply(weight_init)
        for epoch in range(epochs):
            # train
            train_running_loss, train_accurate, valid_running_loss, val_accurate, val_labels, predict_label \
                = train_epoch(net, loss, scheduler, optimizer, train_loader, train_num, batch_size, test_list, test_num)
            train_loss_ls.append(train_running_loss)
            train_accurates.append(train_accurate)
            valid_accurates.append(val_accurate)
            valid_loss_ls.append(valid_running_loss)

            print('[%d Fold epoch %d] train loss: %.4f valide loss: %.4f train_accuracy: %.4f  val_accuracy: %.4f, lr: %.4f, best_val_acc: %.4f'
                % (step + 1, epoch + 1, train_running_loss, valid_running_loss, train_accurate, val_accurate,
                   optimizer.state_dict()['param_groups'][0]['lr'], best_acc))
            if val_accurate > best_acc:
                best_acc = val_accurate
                Y_true = val_labels
                Y_predict = predict_label
                torch.save(
                    {'model_state_dict': net.state_dict(),
                     'best_acc': best_acc},
                    checkpoint_dir)
            log_write.write(str(epoch) + "\t" + str(val_accurate) + "\n")

        train_fold_acc.append(np.max(train_accurates))
        test_fold_acc.append(np.max(valid_accurates))
        valid_accurates = []
        train_accurates = []

        print('%d fold best accuracy: %.4f' % (step + 1, best_acc))
        log_write.write('The average accuracy is: ' + str(np.mean(valid_accurates)) + "\n")
        log_write.write('The best accuracy is: ' + str(best_acc) + "\n")
        best_acc = 0.0
    print('Finishing Training')
    print('average accuracy: %.4f' % np.mean(test_fold_acc))
    x = [i + 1 for i in range(step + 1)]
    plt.figure(figsize=(8, 6))
    # plt.plot(x, train_loss_ls, marker='o', linestyle='--', markersize=5)
    # plt.plot(x, valid_loss_ls, marker='o', linestyle='--', markersize=5)
    # plt.legend(['训练损失', '测试损失'], fontsize=20, loc='best')
    # plt.xlabel('训练次数', fontsize=20)
    # plt.ylabel('损失值', fontsize=20)
    # plt.xticks(fontsize=15)
    # plt.yticks(fontsize=15)
    # x_major_locator = MultipleLocator(100)
    # ax = plt.gca()
    # ax.xaxis.set_major_locator(x_major_locator)
    # plt.show()
    # plt.savefig("../pic/loss%d" % sub + ".svg")

    plt.plot(x, train_fold_acc, marker='x', linestyle='--', markersize=5)
    plt.plot(x, test_fold_acc, marker='x', linestyle='--', markersize=5)
    plt.ylim([0, 1])
    plt.legend(['训练集', '验证集'])
    plt.title('训练结果')
    plt.xlabel('epoch')
    plt.ylabel('准确率')
    x_major_locator = MultipleLocator(100)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.show()
    plt.savefig("../pic/acc %d" % sub + ".svg")

    return best_acc, Y_true, Y_predict

def predict(net, predict_data, true_label, num_layers, num_heads, num_queries, sub):
    net.cpu()
    print(predict_data.dim())
    if predict_data.dim() == 4:
        predict_data = predict_data.cpu()
        output = net(predict_data)
        predict_label = torch.max(output, dim=1)[1]
        attention_weigths = torch.cat(
            [i.reshape((predict_data.shape[0], -1, num_queries, num_queries)).mean(dim=0) for i in net.transformer.attention_weights],
            dim=0).reshape((num_layers, num_heads, -1, num_queries))
        show_heatmaps(attention_weigths.cpu(), xlabel='Key 时间段', ylabel='Query 时间段',
                      titles=['头 %d' % i for i in range(1, 11)], figsize=(12, 9), channel_attention=False, sub=sub)
        # show_heatmaps(net.channel_attention.attention_weights.mean(dim=0, keepdim=True).cpu(), xlabel='Key 通道', ylabel='Query 通道',
        #               titles=['头 %d' % i for i in range(1, 11)], figsize=(12, 9), save_fig=True, channel_attention=True, sub=sub)
        return true_label, predict_label


def main(sub):
    # 参数设置
    nSub = sub
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 72
    epochs = 5
    log_root = './log/kfold_checkpoint_40x200x15_log_subject%d.txt' % nSub
    log_write = open(log_root, 'a')
    net = EEGCNNTransformer(channels=20).cuda()
    # net = EEGCNN_notransformer(channels=20).cuda()
    # net = SpatialTemporalAttention().cuda()
    print('using {} device'.format(device))

    loss = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(net.parameters(), lr=0.0008, betas=(0.5, 0.99), weight_decay=0.0001)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=0.00005)

    root = '../数据集/BCICIV_2a_mat/'

    train_loader_list, train_num, test_num, test_loader_list, _, _ = load_data_kfold(root, sub_num=1, batch_size=72, training=True)
    best_acc, Y_true, Y_predict = train(net, loss, optimizer, scheduler, train_loader_list, test_loader_list, train_num, test_num
                                        , epochs, device, batch_size, log_write, nSub)

    plot_confusion_matrix(Y_true, Y_predict, sub=nSub)

    _, _, _, _, test_data, test_labels = load_data_kfold(root, sub_num=1, batch_size=72, training=True)
    # CNN 6 10 61 STT 3 3 190
    Y_true, Y_predict = predict(net, test_data, test_labels, 6, 10, 30, nSub)
    plot_confusion_matrix(Y_true, Y_predict, sub=nSub)


if __name__ == '__main__':
    for i in range(1):
        main(i + 1)
