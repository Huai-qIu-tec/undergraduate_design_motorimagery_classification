import math
from torch.nn import functional as F
import torch
from torch import nn


class PatchEmbedding(nn.Module):
    def __init__(self, embedding_size):
        super(PatchEmbedding, self).__init__()
        self.embedding_size = embedding_size
        self.embedding = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=5, kernel_size=(1, 51)),
            nn.BatchNorm2d(5),
            nn.LeakyReLU(0.2),
            nn.Conv2d(5, embedding_size, kernel_size=(22, 5), stride=(1, 5)),
        )

    def forward(self, X):
        batch_size = X.shape[0]
        # X.shape = (batch_size, 1, channels, samplepoints)
        out = self.embedding(X)  # out.shape = (batch_size, embedding_size, 1, 128)
        out = out.reshape(batch_size, self.embedding_size, -1)
        out = out.permute(0, 2, 1)  # out.shape = (batch_size, 128, embedding_size)

        return out


class PositionWiseFFN(nn.Module):
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs, dropout, expansion_rate=2):
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


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super(ResidualAdd, self).__init__()
        self.fn = fn

    def forward(self, X, **kwargs):
        residual = X
        out = self.fn(X, **kwargs)
        out += residual
        return out


def transpose_qkv(X, num_heads):
    """
    input:      X.shape = (batch_size，查询或者“键－值”对的个数，num_hiddens)
    output:     X.shape = (batch_size * num_heads, 查询或者“键－值”对的个数, num_hiddens/num_heads)
    """
    # 输?X的形状:(batch_size，查询或者“键－值”对的个数，num_hiddens)
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
        self.num_hiddens = num_hiddens
        self.attention_weights = None
        self.num_heads = num_heads
        self.attention = self.DotProductAttention
        self.dropout = nn.Dropout(dropout)
        self.W_q = nn.Linear(query_size, query_size)
        self.W_k = nn.Linear(key_size, query_size)
        self.W_v = nn.Linear(value_size, query_size)
        self.W_o = nn.Linear(query_size, query_size)

    def DotProductAttention(self, queries, keys, values):
        d = self.num_hiddens
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
    def __init__(self, query_size, key_size, value_size, num_hiddens, num_heads, dropout, ffn_num_input,
                 ffn_num_hiddens, num_layers):
        super(TransformerBlock, self).__init__()
        self.attention_weights = None
        self.num_heads = num_heads

        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(
                "block" + str(i), EncoderBlock(query_size, key_size, value_size, num_hiddens, num_heads, dropout,
                                               ffn_num_input, ffn_num_hiddens))

    def forward(self, X):
        out = X
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            out = blk(out)
            self.attention_weights[i] = blk.attention.attention_weights
        return out


class ChannelAttention(nn.Module):
    def __init__(self, channel_size, dropout=0.3):
        super(ChannelAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.channel_size = channel_size

        # 减少计算
        self.pooling = nn.AvgPool2d(kernel_size=(75, 1), stride=(30, 1))
        self.attention_weights = None
        self.W_q = nn.Sequential(
            nn.LayerNorm(channel_size),
            nn.Linear(channel_size, channel_size),
            # nn.Dropout(dropout)
        )

        self.W_k = nn.Sequential(
            nn.LayerNorm(channel_size),
            nn.Linear(channel_size, channel_size),
            # nn.Dropout(dropout)
        )

        self.W_v = nn.Sequential(
            nn.LayerNorm(channel_size),
            nn.Linear(channel_size, channel_size),
            # nn.Dropout(dropout)
        )

        self.W_o = nn.Sequential(
            nn.LayerNorm(channel_size),
            nn.Linear(channel_size, channel_size),
            # nn.Dropout(dropout)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.weight, 0.0)

    def forward(self, X):# X.shape = (batch_size, 1, channels, samplepoints)

        # X.shape = (batch_szie, 1, samplepoints, channels)
        X = X.permute(0, 1, 3, 2)

        # q, k, v.shape = (batch_szie, 1, samplepoints, channels)
        # query = self.pooling(self.W_q(X)).permute(0, 1, 3, 2)
        # key = self.pooling(self.W_k(X)).permute(0, 1, 3, 2)

        query = self.pooling(self.W_q(X)).permute(0, 1, 3, 2)
        key = self.pooling(self.W_k(X)).permute(0, 1, 3, 2)
        value = self.W_v(X).permute(0, 1, 3, 2)

        self.attention_weights = F.softmax(torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(int(1000 / 30)), dim=-1)

        out = torch.matmul(self.dropout(self.attention_weights), value)
        out = out.permute(0, 1, 3, 2)
        out = self.W_o(out)
        out = out.permute(0, 1, 3, 2)

        return out


class Classification(nn.Module):
    def __init__(self, embedding_size, class_num):
        super(Classification, self).__init__()
        self.embedding_size = embedding_size
        self.fc = nn.Linear(embedding_size, class_num)
        self.ln = nn.LayerNorm(embedding_size)

    def forward(self, X):
        out = torch.mean(X, dim=1)
        out = self.ln(out)
        out = self.fc(out)
        return out


class SpatialTemporalAttention(nn.Module):
    def __init__(self, channel_size=22,
                 query_size=28,
                 key_size=28,
                 value_size=28,
                 num_hiddens=28,
                 num_heads=4,
                 dropout=0.5,
                 ffn_num_input=28,
                 ffn_num_hiddens=28,
                 num_layers=3,
                 class_num=4):
        super(SpatialTemporalAttention, self).__init__()
        self.channel_attention = ChannelAttention(channel_size)
        self.patch_embedding = PatchEmbedding(embedding_size=num_hiddens)
        self.transformer = TransformerBlock(query_size, key_size, value_size, num_hiddens, num_heads, dropout,
                                            ffn_num_input, ffn_num_hiddens, num_layers)
        self.classification = Classification(embedding_size=num_hiddens, class_num=class_num)

    def forward(self, X):
        out = self.channel_attention(X)
        out = self.patch_embedding(out)
        out = self.transformer(out)
        out = self.classification(out)
        return out
