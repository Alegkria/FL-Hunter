import torch as th
import torch.nn.functional as func
import math


class PositionalEncoding(th.nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=15000):
        super(PositionalEncoding, self).__init__()

        self.dropout = th.nn.Dropout(p=dropout)

        pe = th.zeros(max_len, d_model)
        position = th.arange(0, max_len, dtype=th.float).unsqueeze(1)
        div_term1 = th.exp(th.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        div_term2 = th.exp(th.arange(1, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = th.sin(position * div_term1)
        pe[:, 1::2] = th.cos(position * div_term2)

        # pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)  # (n_total_instances, feature_size)


class RelativePositionalEncoding(th.nn.Module):
    def __init__(self, d_model, max_len=15000):
        super(RelativePositionalEncoding, self).__init__()
        self.dropout = th.nn.Dropout(p=0.1)
        self.max_len = max_len

        # 创建相对位置编码矩阵
        self.relative_positions = th.nn.Embedding(2 * max_len - 1, d_model)

    def forward(self, x):
        seq_len = x.size(0)

        # 创建相对位置索引
        positions = th.arange(0, seq_len)
        # positions = positions.unsqueeze(0)

        # 计算相对位置编码
        relative_positions = self.relative_positions(positions)

        # 将相对位置编码与输入相加
        x = x + relative_positions

        return self.dropout(x)


class AttentionWeightPredictor(th.nn.Module):
    def __init__(self,
                 feature_size: int,
                 n_head: int,
                 ):
        super(AttentionWeightPredictor, self).__init__()
        self.feature_size = feature_size
        self.n_head = n_head
        assert feature_size % n_head == 0

        self.relative_position_encoding = RelativePositionalEncoding(feature_size)

        self.w_q = th.nn.Linear(feature_size, feature_size, bias=True)
        self.w_k = th.nn.Linear(feature_size, feature_size, bias=True)
        self.w_v = th.nn.Linear(feature_size, feature_size, bias=True)
        self.fc = th.nn.Linear(feature_size, 1, bias=True)

        self.scale = th.sqrt(th.FloatTensor([feature_size // n_head]))

    def forward(self, query, key, value, mask=None):
        bsz = query.shape[0]
        # m=th.nn.Dropout(p=0.7)
        Q = self.w_q(query)  # [1632,12]  [2,41,3]
        K = self.w_k(key)  # [1632,12]
        V = self.w_v(value)  # [1632,12]

        Q = Q.view(bsz, self.n_head, self.feature_size // self.n_head).permute(0, 1, 2)
        K = K.view(bsz, self.n_head, self.feature_size // self.n_head).permute(0, 1, 2)
        V = V.view(bsz, self.n_head, self.feature_size // self.n_head).permute(0, 1, 2)

        # Q = Q.view(bsz, -1,self.n_head, self.feature_size // self.n_head).permute(0, 2, 1,3)
        # K = K.view(bsz, -1,self.n_head, self.feature_size // self.n_head).permute(0, 2, 1,3)
        # V = V.view(bsz, -1,self.n_head, self.feature_size // self.n_head).permute(0, 2, 1,3)
        # x_with_relative_pos = self.relative_position_encoding(query)

        # print("x_with_relative_pos:",x_with_relative_pos.shape)  # x_with_relative_pos: torch.Size([16, 41, 3])

        scores = th.matmul(Q, K.transpose(-2, -1)) / self.scale
        # print("scores:",scores)
        # print("scores:",scores.size)
        # print("scores:",scores.shape)  # scores: torch.Size([16, 3, 41, 41])

        # scores = scores + x_with_relative_pos

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weight = func.softmax(scores, dim=-1)

        # if dropout is not None:
        # attention_weight = m(attention_weight)

        output = th.matmul(attention_weight, V)
        output = output.permute(0, 1, 2).contiguous()
        # output = output.permute(0, 2,1,3).contiguous()
        output = output.view(bsz, self.n_head * (self.feature_size // self.n_head))
        # output = output.view(bsz, -1,self.n_head * (self.feature_size // self.n_head))

        output = self.fc(output)

        return th.squeeze(output, dim=-1)
        # return self.fc(output)


class PositionwiseFeedForward(th.nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w1 = th.nn.Linear(d_model, d_ff)
        self.w2 = th.nn.Linear(d_ff, d_model)
        self.dropout = th.nn.Dropout(dropout)

    def forward(self, x):
        return self.w2(self.dropout(func.relu(self.w1(x))))


class LayerNorm(th.nn.Module):
    def __init__(self, d_model: int, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = th.nn.Parameter(th.ones(d_model))
        self.beta = th.nn.Parameter(th.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)  # '-1' means last dimension.
        var = x.var(-1, keepdim=True)

        out = (x - mean) / th.sqrt(var + self.eps)
        out = self.gamma * out + self.beta

        return th.squeeze(out, dim=-1)
        # return out