from typing import Literal

from failure_dependency_graph import FDGBaseConfig


# noinspection PyPackageRequirements


class FL_HunterConfig(FDGBaseConfig):
    # training parameters
    early_stopping_epoch_patience: int = 500  # 当模型在验证集上的性能连续500个epoch没有改善时，训练会提前停止

    checkpoint_metric: Literal['val_loss', "MAR"] = "val_loss"  # 选择最佳模型检查点的指标

    # 太小了会导致过拟合，效果下降；但是比较大的时候VAE是无法收敛的
    weight_decay: float = 1e-2  # 权重衰减（L2正则化）的参数，用于控制模型的复杂度，防止过拟合

    ############################
    # FDG
    ############################
    drop_FDG_edges_fraction: float = 0.  # FDG中删除边的比例

    # model parameters
    dropout: bool = False  # 是否使用dropout正则化
    augmentation: bool = False  # 是否进行数据增强

    ################################################
    # Random Walk  p和q: Random Walk算法的两个参数
    p: float = 1 / 4
    q: float = 1 / 4
    random_walk_length: int = 8  # Random Walk的长度

    ###############################################
    # GAT
    GAT_num_heads: int = 1  # GAT（Graph Attention Network）中的头数
    GAT_residual: bool = True  # GAT中是否使用残差连接
    GAT_layers: int = 1  # GAT中的层数
    GAT_shared_feature_mapper: bool = False  # 是否在GAT中共享特征映射器

    ################################################
    # tsfresh tsfresh特征提取模式
    ts_feature_mode: Literal['full', 'simple', 'minimal', 'simple_fctype'] = 'full'

    def configure(self) -> None:  # 配置参数，并添加一些命令行参数
        super().configure()
        self.add_argument("-aug", "--augmentation")
        self.add_argument("-bal", "--balance_train_set")
        self.add_argument("-H", "--GAT_num_heads")
        self.add_argument("-L", "--GAT_layers")
        self.add_argument("-tss", "--train_set_sampling")
# 在该类中，可以通过设置这些属性来调整模型的训练和特征提取的相关参数。
# add_argument()方法用于添加命令行参数，以便在运行代码时可以通过命令行来配置这些参数的值
