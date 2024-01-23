from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Literal

from loguru import logger
from tap import Tap


class FDGBaseConfig(Tap):   # 配置模型训练和特征投影相关的参数
    # Input
    gradient_clip_val: float = 1.  # 梯度裁剪的阈值，默认值为1.0
    es: bool = True  # 是否使用早停止（Early Stopping）策略
    early_stopping_epoch_patience: int = 250  # 早停止策略的阈值，当模型在验证集上的性能连续250个epoch没有改善时，训练会提前停止
    weight_decay: float = 1e-2  # 权重衰减（L2正则化）的参数，默认为1e-2
    init_lr: float = 1e-2  # 初始学习率，默认为1e-2
    max_epoch: int = 3000  # 训练的最大epoch数，默认为3000
    test_second_freq: float = 30.  # 测试（evaluation）的时间间隔，单位为秒，默认为30秒
    test_epoch_freq: int = 100  # 测试的epoch间隔，默认为每100个epoch进行一次测试
    valid_epoch_freq: int = 10  # 验证集评估的epoch间隔，默认为每10个epoch进行一次验证
    display_second_freq: float = 5  # 记录和显示训练信息的时间间隔，单位为秒，默认为每5秒记录一次
    display_epoch_freq: int = 10  # 记录和显示训练信息的epoch间隔，默认为每10个epoch记录一次
    graph_config_path: Optional[Path] = None  # 图结构配置文件的路径，默认为None
    metrics_path: Optional[Path] = None  # 指标配置文件的路径，默认为None
    faults_path: Optional[Path] = None  # 故障信息配置文件的路径，默认为None
    use_anomaly_direction_constraint: bool = False  # 是否使用异常方向约束，默认为False
    data_dir: Path = Path("/SSF/data/aiops2020_phase2/")  # 数据集存储的根目录，默认为/SSF/data/aiops2020_phase2/
    cache_dir: Path = Path('/tmp/SSF/.cache')  # 用本地文件系统能加快速度
    flush_dataset_cache: bool = True  # 是否清空数据集的缓存，默认为

    dataset_split_ratio: Tuple[float, float, float] = (0.4, 0.2, 0.4)  # 表示训练集、验证集和测试集的比例
    train_set_sampling: float = 1.0  # 在训练集中只取前一部分，只用于测试训练集个数对结果的影响的实验
    train_set_repeat: int = 1  # 训练集重复使用的次数，默认为1
    balance_train_set: bool = False  # 是否对训练集进行平衡处理，默认为False

    output_base_path: Path = Path('/SSF/output')  # 输出结果的根目录，默认为/SSF/output
    output_dir: Path = None  # 输出结果的目录，默认为None

    cuda: bool = True  # 是否使用GPU进行计算，默认为True

    ################################################
    # FEATURE PROJECTION
    ################################################
    rec_loss_weight: float = 1.  # 特征投影的重建损失权重，默认为1.0
    FI_feature_dim: int = 3  # 特征投影的输出维度，默认为3
    feature_projector_type: Literal['CNN', 'AE', 'GRU_AE', 'CNN_AE', 'GRU_VAE', 'GRU'] = 'CNN'
    # 特征投影器的类型，默认为'CNN'，可选的类型有['CNN', 'AE', 'GRU_AE', 'CNN_AE', 'GRU_VAE', 'GRU']

    window_size: Tuple[int, int] = (10, 10)  # 窗口大小的设置
    batch_size: int = 16  # 训练时的batch大小，默认为16
    test_batch_size: int = 128  # 测试时的batch大小，默认为128

    def process_args(self) -> None:  # 处理配置参数的设置，设置输出目录和CUDA的使用情况
        if self.output_dir is None:
            import traceback
            caller_file_path = Path(traceback.extract_stack()[-3].filename)
            self.output_dir = Path(
                self.output_base_path / f"{caller_file_path.name}.{datetime.now().isoformat()}"
            ).resolve()
            self.output_dir.mkdir(exist_ok=True, parents=True)
        self.cache_dir.mkdir(exist_ok=True, parents=True)

        import torch
        logger.info(f"{torch.cuda.is_available()=}")
        self.cuda = torch.cuda.is_available() and self.cuda

    def __init__(self, *args, **kwargs):
        super().__init__(*args, explicit_bool=True, **kwargs)
    # 初始化方法，调用父类的构造函数，并设置explicit_bool=True，表示当布尔类型的参数没有显式提供时，默认值为True

    def configure(self) -> None:
        self.add_argument("-z_dim", "--FI_feature_dim")
        self.add_argument("-fe", "--feature_projector_type")
        self.add_argument("-f", "--flush_dataset_cache")
    # 用于配置参数并添加命令行参数，具体的参数配置在这个方法中完成