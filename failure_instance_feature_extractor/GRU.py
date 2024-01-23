import torch as th
from einops import rearrange

from utils.sequential_model_builder import SequentialModelBuilder
from utils.sequential_model_builder import Mulattention


class GRUFeatureModule(th.nn.Module):
    def __init__(self, input_size: th.Size, embedding_size: int, num_layers: int = 1):
        super().__init__()
        self.x_dim = input_size[-2]  # 输入序列的大小
        self.n_ts = input_size[-1]  # 序列中的时间步数
        self.n_instances = input_size[-3]  # 输入数据中的实例数量
        self.z_dim = embedding_size  # 指定的嵌入大小
        self.num_layers = num_layers

        self.encoder = th.nn.GRU(
            input_size=self.x_dim, hidden_size=self.z_dim, num_layers=num_layers,
        )
        # 定义一个 GRU 层作为编码器。它接受每个时间步的 self.x_dim 个输入特征，生成大小为 self.z_dim 的隐藏状态，并具有 num_layers 个叠加的 GRU 层
        self.attention = Mulattention()

        # self.unify_mapper = SequentialModelBuilder(
        #     (-1, self.z_dim, self.n_ts)
        # ).add_flatten(-2).add_linear(128).add_activation().add_linear(embedding_size).build()

        # SequentialModelBuilder 创建一个映射层
        self.unify_mapper = SequentialModelBuilder(
            (-1, self.n_instances, self.z_dim, self.n_ts), debug=False,
        ).add_reshape(
            -1, self.z_dim, self.n_ts,
        ).add_conv_1d(
            out_channels=10, kernel_size=(3,)
        ).add_activation().add_flatten(start_dim=-2).add_linear(embedding_size).add_reshape(
            -1, self.n_instances, embedding_size,
        ).build()
        # 涉及到重塑、1D 卷积、激活函数、扁平化、线性层和重塑

    def forward(self, x):
        z = self.encode(x)
        print("z:", z)
        new_z = self.attention(z, z, z)
        embedding = th.cat([self.unify_mapper(new_z)], dim=-1)
        return embedding

    def encode(self, input_x: th.Tensor) -> th.Tensor:
        """
        :param input_x: (batch_size, n_nodes, n_metrics, n_ts)
        :return: (batch_size, n_nodes, self.z_dim, n_ts)
        """
        batch_size, n_nodes, _, n_ts = input_x.size()
        x = rearrange(input_x, "b n m t -> t (b n) m", b=batch_size, n=n_nodes, m=self.x_dim, t=n_ts)
        # rearrange重新排列张量的维度
        assert x.size() == (n_ts, batch_size * n_nodes, self.x_dim)
        z, _ = self.encoder(x)
        return rearrange(z, "t (b n) z -> b n z t", b=batch_size, n=n_nodes, z=self.z_dim, t=n_ts)
