from functools import reduce
from math import floor
from typing import Union, Tuple, List, Type, Callable, Dict

import dgl
import torch as th
from dgl.nn.pytorch import GATConv
from loguru import logger
from torch import nn as nn

from utils.conv_shape import conv_1d_output_shape, conv_2d_output_shape
from utils.layers import Reshape
from utils.wrapped_graph_nn import WrappedGraphNN


class SequentialModelBuilder:
    def __init__(self, input_shape: Union[th.Size, Tuple[int, ...]], debug=False):
        self._layer_cls: List[Union[Type[nn.Module], Callable[..., nn.Module]]] = []
        self._layer_kwargs: List[Dict] = []
        self._layer_args: List[List] = []
        self._output_shape: List[Tuple[int, ...]] = [input_shape]
        self._debug = debug

    @property
    def output_shapes(self):
        return self._output_shape

    @property
    def last_shape(self):
        return self._output_shape[-1]

    def build(self) -> nn.Module:
        layers = []
        if self._debug:
            logger.debug(f"=============================build layers==================================")
        for _cls, args, kwargs, shape in zip(
                self._layer_cls, self._layer_args, self._layer_kwargs, self._output_shape[1:]
        ):
            if self._debug:
                logger.debug(f"{_cls.__name__} {args=} {kwargs=} {shape=}")
            # noinspection PyArgumentList
            layers.append(_cls(*args, **kwargs))
        return nn.Sequential(
            *layers
        )

    def add_activation(self, activation='gelu') -> 'SequentialModelBuilder':
        if activation.lower() == 'relu':
            self._layer_cls.append(nn.ReLU)
        elif activation.lower() == 'gelu':
            self._layer_cls.append(nn.GELU)
        elif activation.lower() == 'softplus':
            self._layer_cls.append(nn.Softplus)
        else:
            raise RuntimeError(f"{activation=} unknown")
        self._layer_kwargs.append({})
        self._layer_args.append([])
        self._output_shape.append(self._output_shape[-1])
        return self

    def add_linear(
            self, out_features: int, bias: bool = True,
            device=None, dtype=None
    ) -> 'SequentialModelBuilder':
        # out_features 线性层的输出特征数。  bias: 是否使用偏置项，默认为True
        # device: 线性层的计算设备，默认为None  dtype: 线性层的数据类型，默认为None
        self._layer_cls.append(nn.Linear)  # 将nn.Linear类添加到列表中，表示添加了一个线性层
        in_shape = self._output_shape[-1]  # 类内部的一个列表，用于存储每一层的输出形状。[-1]表示取列表中的最后一个元素，即上一个层的输出形状，赋值给in_shape
        self._layer_kwargs.append(
            dict(in_features=in_shape[-1], out_features=out_features, bias=bias,
                 device=device, dtype=dtype)
        )
        self._layer_args.append([])
        self._output_shape.append(in_shape[:-1] + (out_features,))
        return self

    def add_reshape(self, *args) -> 'SequentialModelBuilder':
        self._layer_cls.append(Reshape)
        self._layer_kwargs.append({})
        self._layer_args.append(list(args))
        self._output_shape.append(tuple(args))
        return self

    def add_flatten(self, start_dim: int = 1, end_dim: int = -1) -> 'SequentialModelBuilder':
        self._layer_cls.append(nn.Flatten)
        self._layer_kwargs.append(dict(start_dim=start_dim, end_dim=end_dim))
        self._layer_args.append(list())
        input_shape = self.last_shape
        if end_dim != -1:
            self._output_shape.append(
                input_shape[:start_dim] + (
                    reduce(lambda a, b: a * b, input_shape[start_dim:end_dim + 1], 1),
                ) + input_shape[end_dim + 1:]
            )
        else:
            self._output_shape.append(
                input_shape[:start_dim] + (
                    reduce(lambda a, b: a * b, input_shape[start_dim:], 1),
                )
            )
        return self

    def add_max_pool_1d(self, kernel_size, stride=None, padding=0, dilation=1) -> 'SequentialModelBuilder':
        if stride is None:
            stride = kernel_size
        self._layer_cls.append(nn.MaxPool1d)
        self._layer_kwargs.append(dict(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation))
        self._layer_args.append([])
        self._output_shape.append(
            self.last_shape[:-1] + (
                floor((self.last_shape[-1] + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1),
            )
        )
        return self

    def add_conv_1d(
            self,
            out_channels: int,
            kernel_size: Tuple[int],
            stride=1,
            padding=0,
            bias: bool = True,
    ) -> 'SequentialModelBuilder':
        self._layer_cls.append(nn.Conv1d)
        in_shape = self._output_shape[-1]
        self._layer_kwargs.append(
            dict(
                in_channels=in_shape[-2], out_channels=out_channels, kernel_size=kernel_size,
                stride=stride, padding=padding, bias=bias,
            )
        )
        self._layer_args.append([])
        after_conv_size = conv_1d_output_shape(in_shape[-1], kernel_size, stride=stride, pad=padding)
        self._output_shape.append(in_shape[:-2] + (out_channels, after_conv_size))
        return self

    def add_conv_2d(
            self,
            out_channels: int,
            kernel_size: Tuple[int, int],
            stride=1,
            padding=0,
            bias: bool = True,
    ) -> 'SequentialModelBuilder':
        self._layer_cls.append(nn.Conv2d)
        in_shape = self._output_shape[-1]
        self._layer_kwargs.append(
            dict(
                in_channels=in_shape[-3], out_channels=out_channels, kernel_size=kernel_size,
                stride=stride, padding=padding, bias=bias,
            )
        )
        self._layer_args.append([])
        after_conv_size = conv_2d_output_shape(in_shape[-2:], kernel_size, stride=stride, pad=padding)
        self._output_shape.append(in_shape[:-3] + (out_channels,) + after_conv_size)
        return self

    def add_dropout(self, p: float = 0.5):
        self._layer_cls.append(nn.Dropout)
        self._layer_kwargs.append({})
        self._layer_args.append([p])
        self._output_shape.append(self.last_shape)
        return self

    def add_conv_transpose_1d(
            self,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            output_padding: int = 0
    ):
        self._layer_cls.append(nn.ConvTranspose1d)
        self._layer_kwargs.append(dict(
            in_channels=self.last_shape[-2],
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding
        ))
        self._layer_args.append([])
        self._output_shape.append(
            self.last_shape[:-2] + (out_channels,) + (
                (self.last_shape[-1] - 1) * stride - 2 * padding + (kernel_size - 1) + output_padding + 1,
            )
        )
        return self

    def add_gat_conv_fixed_graph(
            self, graph: dgl.DGLGraph, in_feats: int, out_feats: int, num_heads: int, residual: bool,
            attn_drop: float = 0., feat_drop: float = 0.
    ):
        self._layer_cls.append(lambda *args, **kwargs: WrappedGraphNN(GATConv(*args, **kwargs), graph=graph))
        self._layer_kwargs.append(dict(
            in_feats=in_feats,
            out_feats=out_feats,
            num_heads=num_heads,
            residual=residual,
            attn_drop=attn_drop,
            feat_drop=feat_drop,
        ))
        self._layer_args.append([])
        self._output_shape.append(
            self.last_shape[:-1] + (out_feats * num_heads,)
        )
        return self


class Mulattention(th.nn.Module):
    def __init__(self,
                 feature_size: int,
                 n_head: int,
                 ):
        super(Mulattention, self).__init__()
        self.feature_size = feature_size
        self.n_head = n_head
        assert feature_size % n_head == 0

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
