# -*- coding: UTF-8 -*-

import math
from torch.nn import functional as F
import torch
from torch import nn


class PositionalEmbedding(nn.Module):
    """
    function: 位置编码，这里的位置编码用的可学习的变量，没有用transformer的sin和cos
    input: X
    output: X + pos_embedding
    """

    def __init__(self, channels, embedding_size, dropout):
        super(PositionalEmbedding, self).__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, channels, embedding_size))
        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, X):
        out = X + self.pos_embedding  # .to(X.device)
        if self.dropout:
            out = self.dropout(out)

        return out


class PositionWiseFFN(nn.Module):
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs, dropout, expansion_rate=4):
        super(PositionWiseFFN, self).__init__()
        self.dropout = dropout
        if self.dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
        self.fc1 = nn.Linear(ffn_num_input, ffn_num_hiddens * expansion_rate)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(ffn_num_hiddens * expansion_rate, ffn_num_outputs)


    def forward(self, X):
        out = self.activation(self.fc1(X))
        if self.dropout:
            out = self.dropout(out)
        out = self.fc2(out)
        return out


def transpose_qkv(X, num_heads):
    """
    input:      X.shape = (batch_size，查询或者“键－值”对的个数，num_hiddens)
    output:     X.shape = (batch_size * num_heads, 查询或者“键－值”对的个数, num_hiddens/num_heads)
    """
    # 输⼊X的形状:(batch_size，查询或者“键－值”对的个数，num_hiddens)
    # 输出X的形状:(batch_size，查询或者“键－值”对的个数，num_heads，num_hiddens/num_heads)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)

    # 输出X的形状: (batch_size，num_heads，查询或者“键－值”对的个数, num_hiddens/num_heads)
    X = X.permute(0, 2, 1, 3)

    # 最终输出的形状: (batch_size * num_heads, 查询或者“键－值”对的个数, num_hiddens/num_heads)
    return X.reshape(-1, X.shape[2], X.shape[3])


def transpose_output(X, num_heads):
    """
    input:      X.shape = (batch_size * num_heads, 查询或者“键－值”对的个数, num_hiddens/num_heads)
    output:     X.shape = (batch_size，查询或者“键－值”对的个数，num_hiddens)
    """
    # 输入X.shape = (batch_size * num_heads, 查询或者“键－值”对的个数, num_hiddens/num_heads)
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])

    # 输入X.shape = (batch_size, num_heads, 查询或者“键－值”对的个数, num_hiddens/num_heads)
    X = X.permute(0, 2, 1, 3)

    # 输入X.shape = (batch_size, 查询或者“键－值”对的个数, num_heads, num_hiddens/num_heads)
    return X.reshape(X.shape[0], X.shape[1], -1)


class MultiHeadAttention(nn.Module):
    def __init__(self, query_size, key_size, value_size, num_hiddens, num_heads, dropout):
        super(MultiHeadAttention, self).__init__()
        self.attention_weights = None
        self.num_heads = num_heads
        self.attention = self.DotProductAttention
        self.dropout = nn.Dropout(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens)
        self.W_k = nn.Linear(key_size, num_hiddens)
        self.W_v = nn.Linear(value_size, num_hiddens)
        self.W_o = nn.Linear(num_hiddens, num_hiddens)

    def DotProductAttention(self, queries, keys, values):
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = F.softmax(scores, dim=-1)
        return torch.bmm(self.dropout(self.attention_weights), values)

    def forward(self, X):
        queries = transpose_qkv(self.W_q(X), self.num_heads)
        keys = transpose_qkv(self.W_k(X), self.num_heads)
        values = transpose_qkv(self.W_v(X), self.num_heads)

        # output.shape = (batch_size * num_heads, 查询或者“键－值”对的个数, num_hiddens/num_heads)
        output = self.attention(queries, keys, values)

        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)


class EncoderBlock(nn.Module):
    def __init__(self, query_size, key_size, value_size, num_hiddens, num_heads, dropout, ffn_num_input,
                 ffn_num_hiddens):
        super(EncoderBlock, self).__init__()
        self.norm1 = nn.LayerNorm(num_hiddens)
        self.attention = MultiHeadAttention(query_size, key_size, value_size, num_hiddens, num_heads, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, ffn_num_input, dropout)
        self.norm2 = nn.LayerNorm(num_hiddens)

    def forward(self, X):
        # layer1 MultiHeadAttention + LayerNorm + Residual #
        residual = X
        out = self.norm1(X)
        out = self.attention(out)
        out += residual

        # layer2 PositionalWiseNet + LayerNorm + Residual + LayerNorm + Residual #
        residual = out
        out = self.norm2(out)
        out = self.ffn(out)
        out += residual
        return out


class TransformerBlock(nn.Module):
    def __init__(self, channels, query_size, key_size, value_size, num_hiddens, num_heads, dropout,
                 ffn_num_input, ffn_num_hiddens, num_layers):
        super(TransformerBlock, self).__init__()
        self.attention_weights = None
        self.num_heads = num_heads
        self.pos_embedding = PositionalEmbedding(channels, num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(
                "block" + str(i), EncoderBlock(query_size, key_size, value_size, num_hiddens, num_heads, dropout,
                                               ffn_num_input, ffn_num_hiddens))

    def forward(self, X):
        out = self.pos_embedding(X)
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            out = blk(out)
            self.attention_weights[i] = blk.attention.attention_weights
        return out


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        # self.block1 = nn.Sequential(
        #     nn.Conv2d(in_channels=in_channels, out_channels=40, kernel_size=(1, 25), stride=(1, 1)),
        #     nn.BatchNorm2d(40),
        #     nn.ELU()
        # )
        #
        # self.block2 = nn.Sequential(
        #     nn.Conv2d(in_channels=40, out_channels=40, kernel_size=(16, 1), stride=(1, 1)),
        #     nn.BatchNorm2d(40),
        #     nn.ELU()
        # )
        #
        # self.block3 = nn.Sequential(
        #     nn.AvgPool2d(kernel_size=(1, 75), stride=(1, 15)),
        #     nn.Dropout(0.5)
        # )
        #
        # self.block4 = nn.Sequential(
        #     nn.Conv2d(40, out_channels, kernel_size=(1, 1), stride=(1, 1)),
        #     nn.BatchNorm2d(out_channels)
        # )

        self.shallownet = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=40, kernel_size=(1, 25), stride=(1, 1)),
            nn.Conv2d(in_channels=40, out_channels=40, kernel_size=(22, 1), stride=(1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 75), stride=(1, 15)),
            # pooling acts as slicing to obtain 'patch' along the time dimension as in ViT
            nn.Dropout(0.5),
        )

        self.projection_block = nn.Sequential(
            nn.Conv2d(in_channels=40, out_channels=out_channels, kernel_size=(1, 1), stride=(1, 1))
        )

    def forward(self, X):  # X.shape = (batch_size, 1, channels, samplepoints)
        # 这里的128是num_hiddens
        # out.shape = (batch_size, 128, channels, samplepoints'(64))
        out = self.shallownet(X)

        out = self.projection_block(out).squeeze(2)

        return out.permute(0, 2, 1)


class ClassificationHead(nn.Module):
    def __init__(self):
        super(ClassificationHead, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(2440, 256),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, 4)
        )

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        out = self.fc(x)
        return out


class EEGCNNTransformer(nn.Module):
    def __init__(self,
                 channels,
                 query_size=40,
                 key_size=40,
                 value_size=40,
                 num_hiddens=40,
                 num_heads=10,
                 dropout=0.5,
                 ffn_num_input=40,
                 ffn_num_hiddens=40,
                 num_layers=6,
                 in_channels=1,
                 out_channels=40,
                 num_class=4):
        super(EEGCNNTransformer, self).__init__()
        self.cls_token = nn.Parameter(torch.zeros((1, 1, num_hiddens)), requires_grad=True)
        self.channels = channels
        self.conv_layer = ConvBlock(in_channels=in_channels, out_channels=out_channels)
        self.transformer = TransformerBlock(self.channels, query_size, key_size, value_size, num_hiddens, num_heads,
                                            dropout, ffn_num_input, ffn_num_hiddens, num_layers)
        self.classification = ClassificationHead()

    def forward(self, X):  # X.shape = (batch_size, channels, sampleponits)
        batch_size = X.shape[0]

        # X.shape = (batch_size, 1, channels, samplepoints)

        # out.shape = (batch_size channels, out_channels(128))
        out = self.conv_layer(X)

        # cls_token.shape = (batch_size, 1, num_hiddens)
        # cls_tokens = self.cls_token.repeat(batch_size, 1, 1)

        # out.shape = (batch_size, channels + 1, out_channels)
        # out = torch.cat([cls_tokens, out], dim=1)
        # print(out)
        # out.shape = (batch_size, channels + 1, out_channels)
        print(out.shape)
        out = self.transformer(out)
        print(out.shape)
        # logits.shape = (batch_size, 4)
        # logits = self.fc(out[:, 0, :])

        logits = self.classification(out)

        return logits