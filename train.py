import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as plt
from model.Spatial_Temporal_Attention import SpatialTemporalAttention
from model.CNNTransformer import EEGCNNTransformer
from tools.utils import load_data, aug_data, plot_confusion_matrix, show_heatmaps
from torchsummary import summary
from model.CNNTransformer_notransformer import EEGCNN_notransformer
from sklearn.metrics import confusion_matrix

plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']


def train_epoch(net, loss, scheduler, optimizer, train_loader, train_num, batch_size):
    running_loss = 0.0
    train_steps = len(train_loader)
    acc = 0.0

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
        running_loss += l.sum().item() * signals.size(0)
    train_accurate = acc / (train_num * 2)
    return running_loss / (len(train_loader.dataset) * 2), train_steps, train_accurate


def validate(net, test_loader, loss, test_num, device):
    net.eval()
    test_steps = len(test_loader)
    acc = 0.0

    running_loss = 0.0

    val_labels_ls = torch.Tensor().cuda()
    predict_labels_ls = torch.Tensor().cuda()
    with torch.no_grad():
        for step, val_data in enumerate(test_loader):
            val_signals, val_labels = val_data
            val_signals = val_signals.cuda()
            val_labels = val_labels.long().cuda()
            outputs = net(val_signals)
            predict_labels = torch.max(outputs, dim=1)[1]
            l = loss(outputs, val_labels)

            acc += torch.eq(predict_labels, val_labels).sum().item()
            val_labels_ls = torch.cat([val_labels_ls, val_labels])
            predict_labels_ls = torch.cat([predict_labels_ls, predict_labels])
            running_loss += l.sum().item() * val_signals.size(0)
    val_accurate = acc / test_num
    cm = confusion_matrix(val_labels.cpu(), predict_labels.cpu())
    pe = [cm[i, i] * cm[i, :].sum() for i in range(4)]
    pe = np.sum(pe) / (val_labels.shape[0] ** 2)
    kappa = (val_accurate - pe) / (1 - pe)
    return running_loss / len(test_loader.dataset), test_steps, val_accurate, val_labels_ls, predict_labels_ls, kappa


def train(net, loss, optimizer, scheduler, train_loader, test_loader, train_num, test_num, save_path, epochs, device,
          batch_size, log_write, sub, checkpoint=None):
    if checkpoint is None:
        best_acc = 0.0
        kappa = 0.0
    else:
        best_acc = checkpoint['best_acc']
        kappa = checkpoint['kappa']
    Y_true = 0.0
    Y_predict = 0.0
    train_accurates = []
    valid_accurates = []
    train_loss_ls = []
    valid_loss_ls = []

    for epoch in range(epochs):
        # train
        train_running_loss, train_steps, train_accurate = train_epoch(net, loss, scheduler, optimizer, train_loader,
                                                                      train_num, batch_size)
        train_loss_ls.append(train_running_loss)
        train_accurates.append(train_accurate)

        # validate
        valid_running_loss, test_steps, val_accurate, val_labels, predict_label, k = validate(net, test_loader, loss,
                                                                                           test_num, device)
        valid_accurates.append(val_accurate)
        valid_loss_ls.append(valid_running_loss)

        print('[epoch %d] train loss: %.4f valide loss: %.4f train_accuracy: %.4f  val_accuracy: %.4f, lr: %.4f, best_val_acc: %.4f, kappa: %.4f'
            % (epoch + 1, train_running_loss, valid_running_loss, train_accurate, val_accurate,
               optimizer.state_dict()['param_groups'][0]['lr'], best_acc, kappa))
        if val_accurate > best_acc:
            kappa = k
            best_acc = val_accurate
            Y_true = val_labels
            Y_predict = predict_label
            torch.save(
                {'model_state_dict': net.state_dict(),
                 'best_acc': best_acc,
                 'kappa': kappa},
                save_path)
        log_write.write(str(epoch) + "\t" + str(val_accurate) + "\n")

    print('Finishing Training')
    print('best accuracy: %.3f' % best_acc)
    log_write.write('The average accuracy is: ' + str(np.mean(valid_accurates)) + "\n")
    log_write.write('The best accuracy is: ' + str(best_acc) + "\n")

    x = [i + 1 for i in range(epochs)]
    plt.figure(figsize=(8, 6))
    plt.plot(x, train_loss_ls, marker='o', linestyle='--', markersize=5)
    plt.plot(x, valid_loss_ls, marker='o', linestyle='--', markersize=5)
    plt.legend(['训练损失', '测试损失'], fontsize=20, loc='best')
    plt.xlabel('训练次数', fontsize=20)
    plt.ylabel('损失值', fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    x_major_locator = MultipleLocator(200)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.show()
    plt.savefig("../pic/loss%d" % sub + ".svg")

    plt.plot(x, train_accurates, marker='x', linestyle='--', markersize=5)
    plt.plot(x, valid_accurates, marker='x', linestyle='--', markersize=5)
    plt.ylim([0, 1])
    plt.legend(['训练集', '验证集'])
    plt.title('训练结果')
    plt.xlabel('epoch')
    plt.ylabel('准确率')
    x_major_locator = MultipleLocator(200)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
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
            [i.reshape((288, -1, num_queries, num_queries)).mean(dim=0) for i in net.transformer.attention_weights],
            dim=0).reshape((num_layers, num_heads, -1, num_queries))
        show_heatmaps(attention_weigths.cpu(), xlabel='Key 时间段', ylabel='Query 时间段',
                      titles=['头 %d' % i for i in range(1, 11)], figsize=(12, 9), channel_attention=False, sub=sub)
        show_heatmaps(net.channel_attention.attention_weights.mean(dim=0, keepdim=True).cpu(), xlabel='Key 通道', ylabel='Query 通道',
                      titles=['头 %d' % i for i in range(1, 11)], figsize=(12, 9), save_fig=True, channel_attention=True, sub=sub)
        return true_label, predict_label


def set_rng_seed(seed):
    random.seed(seed) #为python设置随机种子
    np.random.seed(seed)  #为numpy设置随机种子
    torch.manual_seed(seed)   #为CPU设置随机种子
    torch.cuda.manual_seed(seed)   #为当前GPU设置随机种子
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed_all(seed)   #为所有GPU设置随机种子
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    torch.use_deterministic_algorithms(True)

def main(args):
    # 参数设置
    set_rng_seed(111)
    nSub = args.nSub
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size
    epochs = args.epochs
    checkpoint = None
    log_root = args.log_root % nSub
    log_write = open(log_root, 'a')
    net = EEGCNNTransformer(channels=args.channels,
                            query_size=args.query_size,
                            key_size=args.query_size,
                            value_size=args.query_size,
                            num_hiddens=args.query_size,
                            num_heads=args.num_heads,
                            dropout=args.dropout,
                            ffn_num_input=args.query_size,
                            ffn_num_hiddens=args.query_size,
                            num_layers=args.num_layers,
                            out_channels=args.query_size
                            ).cuda()
    # net = SpatialTemporalAttention().cuda()
    print('using {} device'.format(device))
    import torchsummary
    print(torchsummary.summary(net, (1, 22, 1000)))
    # load 模型参数
    checkpoint_dir = args.checkpoint_root % nSub
    if os.path.exists(checkpoint_dir):
        checkpoint = torch.load(checkpoint_dir)
        model_state_dict = checkpoint['model_state_dict']
        # metrics = [checkpoint['best_acc'], checkpoint['kappa']]
        net.load_state_dict(model_state_dict)

    loss = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.5, 0.99), weight_decay=0.0001)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500, eta_min=0.00001)

    root = '../数据集/BCICIV_2a_mat/'

    if args.train:
        train_loader, test_loader, train_num, test_num, _, _ = load_data(root, sub_num=nSub, batch_size=batch_size, training=True)
        best_acc, Y_true, Y_predict = train(net, loss, optimizer, scheduler, train_loader, test_loader, train_num, test_num,
                                            checkpoint_dir, epochs, device, batch_size, log_write, nSub, checkpoint)
        plot_confusion_matrix(Y_true, Y_predict, sub=nSub)
    else:
        _, _, _, _, test_data, test_labels = load_data(root, sub_num=nSub, batch_size=batch_size, training=True)
        # CNN 6 10 61 STT 3 3 190
        Y_true, Y_predict = predict(net, test_data, test_labels, args.num_layers, args.num_heads, args.channels, nSub)
        plot_confusion_matrix(Y_true, Y_predict, sub=nSub)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='cnn-transformer')
    parser.add_argument('--nSub', default=1, type=int, help='subject')
    parser.add_argument('--channels', default=20, type=int, help='positional parameter')
    parser.add_argument('--query_size', default=40, type=int, help='query size')
    parser.add_argument('--lr', default=5e-4, type=float, help='learning rate')
    parser.add_argument('--num_heads', default=10, type=int, help='self-attention head nums')
    parser.add_argument('--num_layers', default=6, type=int, help='transformer block nums')
    parser.add_argument('--dropout', default=0.5, type=float, help='dropout rate')
    parser.add_argument('--batch_size', default=72, type=int, help='batch size')
    parser.add_argument('--epochs', default=2000, type=int, help='train epochs')
    parser.add_argument('--log_root', default='./log/EEGformer_log_subject%d.txt', type=str, help='log root')
    parser.add_argument('--checkpoint_root', default='./model_state_dict/conformer/conformer_sub%d.pth', type=str,
                        help='checkpoint root')
    parser.add_argument('--train', default=True, action='store_true')
    args = parser.parse_args()
    main(args)
